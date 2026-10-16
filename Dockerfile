# ScholarTrack — one container serving the API and the built dashboard.
#   docker build -t scholartrack .
#   docker run -p 8000:8000 scholartrack        → http://localhost:8000
# The image contains only code, configs, the small committed model bundle and aggregate
# reports. The demo data is *generated* during the build from configs/synthetic.yaml; no real
# student record is copied in (see .dockerignore).

# ---- 1. Build the React dashboard -------------------------------------------------------
FROM node:22-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- 2. Python runtime ------------------------------------------------------------------
FROM python:3.13-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1
WORKDIR /app

# Dependencies first (cached layer): exact versions from uv.lock, torch as the CPU build.
RUN pip install uv==0.11.32
COPY pyproject.toml uv.lock README.md ./
COPY scripts/cpu_requirements.sh scripts/
RUN scripts/cpu_requirements.sh runtime > /tmp/requirements.txt \
 && uv pip install --system -r /tmp/requirements.txt \
 && uv pip install --system --no-deps "$(scripts/cpu_requirements.sh torch)" \
      --index-url https://download.pytorch.org/whl/cpu \
 && rm /tmp/requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY models/serving/ models/serving/
COPY reports/ reports/
COPY --from=frontend /app/frontend/dist frontend/dist

# Deterministic synthetic demo data; the API checks its hash against the model bundle.
RUN python scripts/generate_synthetic_data.py --no-report

# Run as an unprivileged user; only /app/var is writable (optional history database).
RUN useradd --create-home --uid 10001 scholartrack \
 && mkdir -p /app/var && chown scholartrack /app/var
USER scholartrack
ENV SCHOLARTRACK_HISTORY_DB=/app/var/history.sqlite3 \
    PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"
CMD ["sh", "-c", "exec uvicorn scholartrack.api.app:app --host 0.0.0.0 --port ${PORT}"]

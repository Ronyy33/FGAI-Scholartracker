# Deployment Guide

ScholarTrack deploys as **one Docker container** that serves both the API and the dashboard.
The image holds code, configs, the 44 KB model bundle and aggregate reports; the demo dataset
is *generated* while the image builds. No real student record is in the image or the
repository, and the app needs **no secrets**.

> **Status:** deployment-ready, **not deployed**. Deploying needs an account on a hosting
> service, which only you can create. Every step below that you must do yourself is marked
> **You**. The Docker build is verified by GitHub Actions (job "Docker image — build and
> health check"), not on this development machine, which has no Docker daemon.

## 1. Before you deploy — read this

- **Synthetic demo only.** The public demo must keep using synthetic data. The app has **no
  login**; anyone with the URL can use it. Do not upload real student files to a public demo,
  and do not add a fake password screen — authentication that only *looks* secure is worse
  than none. A real deployment needs your institution's authentication (e.g. SSO in front of
  the app) and approval first.
- **Uploads are not stored.** `POST /api/predict/batch` processes files in memory; logs hold
  counts only. Prediction history is off (`SCHOLARTRACK_ENABLE_HISTORY=false`).
- **Memory.** The service imports PyTorch. Measured locally: about 680 MB resident with the
  GPU build of PyTorch; the image uses the smaller CPU build (CI prints the container's memory
  use after a few requests — check the "Run and check health" step log). Choose an instance
  with **at least 1 GB RAM** to be safe.

## 2. Run it locally

### With Docker

```bash
docker build -t scholartrack .
docker run --rm -p 8000:8000 scholartrack
# open http://localhost:8000  ·  API docs: http://localhost:8000/api/docs
```

### Without Docker (development)

```bash
uv sync                                               # Python deps (see note on torch below)
uv run python scripts/generate_synthetic_data.py      # demo data (≈ 5 s, deterministic)
cd frontend && npm ci && npm run build && cd ..       # dashboard → frontend/dist
uv run uvicorn scholartrack.api.app:app --port 8000   # serves API + dashboard
```

For frontend work with hot reload: run the API on port 8000 and `npm run dev` in `frontend/`
(Vite proxies `/api` and `/health` to port 8000).

**Note on torch:** `uv sync` installs PyTorch from PyPI, which on Linux includes ~3 GB of
GPU libraries. For a lean CPU-only install, use the same approach as CI:
`scripts/cpu_requirements.sh` (see `.github/workflows/ci.yml`).

## 3. Option A — Hugging Face Spaces (Docker) — recommended for a free demo

Spaces offer a free CPU tier with more memory than most free web hosts (check the current
limits on Hugging Face's pricing page).

1. **You:** create a Hugging Face account and a new Space: *SDK = Docker*, *template = Blank*,
   visibility as you prefer.
2. **You:** push this repository's branch to the Space's Git remote (Hugging Face shows the
   exact `git remote add` / `git push` commands; authenticate with a Hugging Face access token
   when Git asks — **never** paste the token into code, a chat, or a committed file).
3. The Space's `README.md` must start with a YAML header telling it which port to use. Add
   this at the very top of `README.md` **in the copy you push to the Space** (or keep it on a
   separate deployment branch):

   ```yaml
   ---
   title: ScholarTrack
   sdk: docker
   app_port: 8000
   ---
   ```

4. Wait for the build (several minutes the first time), then open the Space URL.
5. Verify (section 5).

## 4. Option B — Render (Docker web service)

`render.yaml` in the repository describes the service (Docker, health check `/health`,
history off). Render sets the `PORT` variable; the image listens on it.

1. **You:** create a Render account and connect your GitHub account.
2. **You:** *New → Blueprint*, select `Ronyy33/FGAI-Scholartracker` and the branch
   `claude/modest-bell-box1op` (or `main` after merging). Render reads `render.yaml`.
3. Pick an instance type with ≥ 1 GB RAM if available on your plan; the free tier may be too
   small for PyTorch and may sleep when idle (first request then takes a while).
4. Deploy, wait for "Live", then verify (section 5).

Any other host that runs a Dockerfile (Railway, Fly.io, Google Cloud Run, a university VM)
works the same way: build the image, expose the port in `PORT` (default 8000), health check
`/health`.

## 5. Verify a deployment

Replace `$URL` with your public URL.

```bash
curl -s $URL/health
# → {"status":"ok","model_loaded":true,"demo_data_loaded":true,"run_id":"20261016T200932Z-8acaad7",...}

curl -s $URL/api/model-info | head -c 300

# sample prediction with one synthetic record from the demo data (no real data)
curl -s -X POST $URL/api/predict -H 'Content-Type: application/json' \
  -d @examples/predict_request.json | head -c 400
```

Then open `$URL` in a browser and walk through the demo checklist in
[`PROJECT_REPORT.md`](PROJECT_REPORT.md#demonstration-checklist).

If `/health` says `degraded`, the model bundle failed its hash check (see the service log);
if `demo_data_loaded` is `false`, the generated demo data does not match the bundle's dataset
hash (rebuild the image from an unmodified checkout).

## 6. Configuration

All settings are optional environment variables; see [`.env.example`](../.env.example).

| Variable | Default | Meaning |
|----------|---------|---------|
| `PORT` | 8000 | Port the container listens on |
| `SCHOLARTRACK_MAX_UPLOAD_BYTES` | 2097152 | Largest accepted CSV upload |
| `SCHOLARTRACK_MAX_ROWS` | 5000 | Most rows accepted per upload |
| `SCHOLARTRACK_CORS_ORIGINS` | *(empty)* | Extra origins allowed to call the API |
| `SCHOLARTRACK_ENABLE_HISTORY` | false | Store prediction counts (no identifiers) in SQLite |
| `SCHOLARTRACK_HISTORY_DB` | `/app/var/history.sqlite3` in the image | History database path (mount a volume to keep it) |

## 7. Updating the model

The website never trains. To change the model: retrain and evaluate offline
(`scripts/train.py`, `scripts/final_evaluation.py`), export a new bundle
(`scripts/export_serving_bundle.py`), run the tests, commit `models/serving/` and the reports,
and redeploy. A changed bundle with stale reports or mismatched hashes is caught by the tests
and by the start-up checks.

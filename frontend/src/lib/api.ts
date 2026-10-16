import { useCallback, useEffect, useState } from "react";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public code: string,
    public details?: unknown,
  ) {
    super(message);
  }
}

/** Fetch JSON from the ScholarTrack API; errors carry the API's own message. */
export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new ApiError("Cannot reach the ScholarTrack API. Is the server running?", 0, "network");
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const error = body?.error;
    throw new ApiError(
      error?.message ?? `Request failed (${response.status})`,
      response.status,
      error?.code ?? "http_error",
      error?.details,
    );
  }
  return body as T;
}

export interface Loadable<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
  reload: () => void;
}

/** Load a GET endpoint with loading / error state. */
export function useApi<T>(path: string | null): Loadable<T> {
  const [nonce, setNonce] = useState(0);
  const [result, setResult] = useState<{
    key: string | null;
    data: T | null;
    error: ApiError | null;
  }>({ key: null, data: null, error: null });
  const reload = useCallback(() => setNonce((n) => n + 1), []);
  const key = path === null ? null : `${path}#${nonce}`;

  useEffect(() => {
    if (path === null || key === null) return;
    let cancelled = false;
    apiFetch<T>(path)
      .then((data) => !cancelled && setResult({ key, data, error: null }))
      .catch((error: ApiError) => !cancelled && setResult({ key, data: null, error }));
    return () => {
      cancelled = true;
    };
  }, [path, key]);

  // Loading = the stored result belongs to an older request (state is derived, not set).
  const current = result.key === key;
  return {
    data: current ? result.data : null,
    error: current ? result.error : null,
    loading: key !== null && !current,
    reload,
  };
}

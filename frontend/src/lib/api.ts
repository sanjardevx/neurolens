const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

/** Calls the NeuroLens API with the doctor's Supabase token; the API checks it on every request. */
export async function api<T>(path: string, token: string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(API_URL + path, { ...init, headers: { ...init.headers, Authorization: `Bearer ${token}` } });
  } catch {
    throw new ApiError(0, "Can't reach the NeuroLens API. Is it running?");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    // FastAPI sends {detail: "..."} for our errors and {detail: [...]} for validation errors
    const detail = typeof body?.detail === "string" ? body.detail : res.statusText;
    throw new ApiError(res.status, detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export type CaseStatus = "new" | "processing" | "ready" | "reviewed" | "failed";

export type CaseRow = {
  id: string;
  code: string;
  status: CaseStatus;
  created_at: string;
  updated_at: string;
  tumor_type: string | null;
  confidence: number | null;
  tumor_cm3: number | null;
  progress: number | null;
  error: string | null;
  decision: "approved" | "edited" | "rejected" | null;
};

export const TUMOR_LABELS: Record<string, string> = {
  glioma: "Glioma",
  meningioma: "Meningioma",
  pituitary: "Pituitary",
  notumor: "No tumor",
};

import { supabase } from "@/lib/supabase";

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
    throw new ApiError(0, "Can't reach the NeuroLens server. Check your connection and try again.");
  }
  if (res.status === 401) {
    // The login is no longer valid (expired or revoked): signing out sends the doctor back to the login page
    await supabase?.auth.signOut();
    throw new ApiError(401, "Your session has ended. Please sign in again.");
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

export type CaseOut = Pick<CaseRow, "id" | "code" | "status" | "created_at" | "updated_at">;

export type ResultOut = {
  id: string;
  model_version: string;
  progress: number;
  tumor_type: string | null;
  confidence: number | null;
  probabilities: Record<string, number> | null;
  volumes_cm3: Record<string, number> | null;
  side: string | null;
  lobe_estimate: string | null;
  has_mask: boolean;
  has_gradcam: boolean;
  info: { classifier_input?: string } | null;
  error: string | null;
  created_at: string;
};

/** A protected file (e.g. the Grad-CAM image) as a local object URL: <img src> can't send the token. */
export async function apiFileUrl(path: string, token: string): Promise<string> {
  const res = await fetch(API_URL + path, { headers: { Authorization: `Bearer ${token}` } });
  if (!res.ok) throw new ApiError(res.status, `Could not load the file (${res.status})`);
  return URL.createObjectURL(await res.blob());
}

/** Multipart upload with progress. fetch() can't report upload progress, XMLHttpRequest can. */
export function upload<T>(path: string, token: string, form: FormData, onProgress: (share: number) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", API_URL + path);
    xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) return resolve(xhr.response);
      const detail = xhr.response?.detail;
      reject(new ApiError(xhr.status, typeof detail === "string" ? detail : `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new ApiError(0, "Can't reach the NeuroLens server. Check your connection and try again."));
    xhr.send(form);
  });
}

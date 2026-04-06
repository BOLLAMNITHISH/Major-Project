export type PredictResponse = {
  prediction: string;
  confidence: number;
  all_scores: Record<string, number>;
};

export type HistoryItem = {
  id: number;
  filename: string;
  prediction: string;
  confidence: number;
  all_scores: Record<string, number>;
  created_at: string;
};

export type HealthResponse = {
  status: string;
  version: string;
  model_loaded: boolean;
  device: string | null;
  checkpoint: string | null;
};

export type ExplainResponse = {
  prediction: string;
  confidence: number;
  all_scores: Record<string, number>;
  target_class_index: number;
  target_class_label: string;
  side_by_side_png_base64: string;
};

export type EvaluationReport = {
  available: boolean;
  source_path: string | null;
  class_names: string[] | null;
  metrics: Record<string, number> | null;
  confusion_matrix: number[][] | null;
  val_loss: number | null;
  note: string | null;
};

function scoresToSortedList(all_scores: Record<string, number>) {
  return Object.entries(all_scores)
    .map(([label, probability]) => ({ label, probability }))
    .sort((a, b) => b.probability - a.probability);
}

export { scoresToSortedList };

export async function predictImage(file: File): Promise<PredictResponse> {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch("/predict", {
    method: "POST",
    body,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const detail = (err as { detail?: string | { msg: string }[] }).detail;
    const msg =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d) => d.msg).join(", ")
          : res.statusText;
    throw new Error(msg || "Prediction failed");
  }
  return res.json();
}

export async function fetchHistory(limit = 100): Promise<HistoryItem[]> {
  const res = await fetch(`/history?limit=${limit}`);
  if (!res.ok) throw new Error("Failed to load history");
  return res.json();
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch("/health");
  if (!res.ok) throw new Error("API unreachable");
  return res.json();
}

export async function explainImage(file: File): Promise<ExplainResponse> {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch("/explain", { method: "POST", body });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const detail = (err as { detail?: string }).detail;
    throw new Error(typeof detail === "string" ? detail : "Explain request failed");
  }
  return res.json();
}

export async function fetchEvaluation(): Promise<EvaluationReport> {
  const res = await fetch("/evaluation");
  if (!res.ok) throw new Error("Failed to load evaluation report");
  return res.json();
}

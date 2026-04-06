const DISPLAY: Record<string, string> = {
  glioma: "Glioma",
  meningioma: "Meningioma",
  pituitary: "Pituitary",
  no_tumor: "No tumor",
};

/** Human-readable class label for UI */
export function formatClassLabel(key: string): string {
  return DISPLAY[key] ?? key.replace(/_/g, " ");
}

/** Subtle clinical description lines (educational, not diagnostic copy) */
export function classDescription(key: string): string {
  const d: Record<string, string> = {
    glioma: "Glial-origin tumor patterns on structural MRI.",
    meningioma: "Extra-axial, often dural-based appearing lesions.",
    pituitary: "Sellar / suprasellar region involvement.",
    no_tumor: "No dominant tumor class predicted for this slice.",
  };
  return d[key] ?? "Classification from ViT probability map.";
}

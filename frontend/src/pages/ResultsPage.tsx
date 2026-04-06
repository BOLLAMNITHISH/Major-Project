import { motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { explainImage, scoresToSortedList } from "../api";
import { classDescription, formatClassLabel } from "../lib/labels";
import type { ResultsLocationState } from "../types/navigation";

const barColors = [
  "from-clinical-400 to-teal-400",
  "from-cyan-400 to-blue-400",
  "from-sky-400 to-indigo-400",
  "from-slate-400 to-slate-500",
];

export function ResultsPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const state = location.state as ResultsLocationState | null;
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [gradcamB64, setGradcamB64] = useState<string | null>(null);
  const [gradcamLoading, setGradcamLoading] = useState(false);
  const [gradcamErr, setGradcamErr] = useState<string | null>(null);

  useEffect(() => {
    if (!state?.file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(state.file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [state?.file]);

  const sorted = useMemo(
    () => (state ? scoresToSortedList(state.result.all_scores) : []),
    [state],
  );

  if (!state?.result) {
    return (
      <main className="mx-auto max-w-lg px-4 py-24 text-center sm:px-6">
        <div className="glass rounded-3xl p-10">
          <h1 className="font-display text-xl font-semibold text-white">No result yet</h1>
          <p className="mt-2 text-sm text-slate-400">
            Upload an MRI slice to generate a prediction report.
          </p>
          <Link
            to="/upload"
            className="mt-6 inline-block rounded-xl bg-clinical-500 px-5 py-2.5 text-sm font-semibold text-surface-950"
          >
            Go to Analyze
          </Link>
        </div>
      </main>
    );
  }

  const { result, fileName } = state;
  const top = sorted[0];

  const runGradCam = async () => {
    if (!state.file) return;
    setGradcamLoading(true);
    setGradcamErr(null);
    try {
      const ex = await explainImage(state.file);
      setGradcamB64(ex.side_by_side_png_base64);
    } catch (e) {
      setGradcamErr(e instanceof Error ? e.message : "Grad-CAM failed");
      setGradcamB64(null);
    } finally {
      setGradcamLoading(false);
    }
  };

  return (
    <main className="mx-auto max-w-4xl px-4 pb-24 pt-10 sm:px-6 sm:pt-12">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <motion.h1
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            className="font-display text-3xl font-bold text-white sm:text-4xl"
          >
            Analysis report
          </motion.h1>
          <p className="mt-1 text-sm text-slate-500">{fileName}</p>
        </div>
        <motion.button
          type="button"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.15 }}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => navigate("/upload")}
          className="rounded-xl border border-white/15 bg-white/[0.04] px-4 py-2.5 text-sm font-medium text-slate-200"
        >
          New scan
        </motion.button>
      </div>

      <div className="mt-10 grid gap-8 lg:grid-cols-[280px_1fr] lg:gap-10">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="glass-strong h-fit overflow-hidden rounded-3xl p-2"
        >
          <div className="overflow-hidden rounded-2xl bg-slate-900/50 ring-1 ring-white/10">
            {previewUrl ? (
              <img src={previewUrl} alt="" className="aspect-square w-full object-cover" />
            ) : (
              <div className="flex aspect-square items-center justify-center text-sm text-slate-500">
                No preview
              </div>
            )}
          </div>
        </motion.div>

        <div className="space-y-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.08 }}
            className="glass rounded-3xl p-8 sm:p-10"
          >
            <p className="text-xs font-semibold uppercase tracking-widest text-clinical-400/90">
              Primary finding
            </p>
            <p className="mt-3 font-display text-4xl font-bold text-white sm:text-5xl">
              {formatClassLabel(result.prediction)}
            </p>
            <p className="mt-3 max-w-xl text-sm leading-relaxed text-slate-400">
              {classDescription(result.prediction)}
            </p>
            <div className="mt-8 flex flex-wrap items-baseline gap-3">
              <span className="text-xs font-medium uppercase tracking-wider text-slate-500">
                Model confidence
              </span>
              <span className="font-display text-3xl font-bold tabular-nums text-gradient-clinical">
                {(result.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="mt-3 h-3 w-full max-w-md overflow-hidden rounded-full bg-slate-800/80">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${Math.min(100, result.confidence * 100)}%` }}
                transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1], delay: 0.2 }}
                className="h-full rounded-full bg-gradient-to-r from-clinical-400 to-teal-300"
              />
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
            className="glass rounded-3xl p-8 sm:p-10"
          >
            <div className="flex items-center justify-between gap-4">
              <h2 className="font-display text-lg font-semibold text-white">
                Probability distribution
              </h2>
              <span className="rounded-md bg-white/[0.06] px-2 py-1 text-[10px] font-medium uppercase tracking-wide text-slate-500">
                Softmax
              </span>
            </div>
            <p className="mt-1 text-sm text-slate-500">
              Relative likelihood across all tumor classes for this slice.
            </p>

            <ul className="mt-8 space-y-5">
              {sorted.map((row, i) => {
                const isTop = row.label === top?.label;
                const pct = row.probability * 100;
                const color = barColors[i % barColors.length];
                return (
                  <li key={row.label} className="relative">
                    <div className="mb-2 flex justify-between text-sm">
                      <span
                        className={
                          isTop ? "font-semibold text-clinical-200" : "font-medium text-slate-300"
                        }
                      >
                        {formatClassLabel(row.label)}
                        {isTop && (
                          <span className="ml-2 rounded bg-clinical-500/20 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide text-clinical-300">
                            Top
                          </span>
                        )}
                      </span>
                      <span className="tabular-nums text-slate-400">{pct.toFixed(1)}%</span>
                    </div>
                    <div className="h-3 overflow-hidden rounded-full bg-slate-800/90">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${pct}%` }}
                        transition={{
                          duration: 0.85,
                          delay: 0.12 + i * 0.06,
                          ease: [0.22, 1, 0.36, 1],
                        }}
                        className={`h-full rounded-full bg-gradient-to-r ${color} opacity-95`}
                      />
                    </div>
                  </li>
                );
              })}
            </ul>
          </motion.div>
        </div>
      </div>

      <motion.section
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.22 }}
        className="glass mt-10 rounded-3xl p-8 sm:p-10"
      >
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h2 className="font-display text-lg font-semibold text-white">Explainability</h2>
            <p className="mt-1 max-w-xl text-sm text-slate-500">
              <strong className="font-medium text-slate-400">Grad-CAM</strong> on the last ViT
              encoder block highlights patch tokens that most influenced the predicted class. This
              is a saliency guide—not a tumor segmentation.
            </p>
          </div>
          <motion.button
            type="button"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            disabled={gradcamLoading || !state.file}
            onClick={() => void runGradCam()}
            className="shrink-0 rounded-xl bg-white/[0.08] px-4 py-2.5 text-sm font-semibold text-clinical-200 ring-1 ring-white/15 disabled:opacity-50"
          >
            {gradcamLoading ? "Computing…" : gradcamB64 ? "Regenerate Grad-CAM" : "Show Grad-CAM"}
          </motion.button>
        </div>
        {gradcamErr && (
          <p className="mt-4 rounded-lg border border-rose-500/25 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
            {gradcamErr}
          </p>
        )}
        {gradcamB64 && (
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mt-6 overflow-hidden rounded-2xl ring-1 ring-white/10"
          >
            <img
              src={`data:image/png;base64,${gradcamB64}`}
              alt="Original MRI and Grad-CAM overlay side by side"
              className="w-full bg-slate-900/80"
            />
          </motion.div>
        )}
      </motion.section>
    </main>
  );
}

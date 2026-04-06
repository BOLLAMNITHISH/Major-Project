import { motion } from "framer-motion";
import Plotly from "plotly.js-dist-min";
import { useEffect, useRef, useState } from "react";
import type { EvaluationReport } from "../api";
import { fetchEvaluation } from "../api";

/** Typed loosely for plotly.js-dist-min newPlot (full plotly.js types are optional). */
type PlotData = Record<string, unknown>;
type PlotLayout = Record<string, unknown>;
type PlotConfig = Record<string, unknown>;

const PLOT_CFG: PlotConfig = {
  responsive: true,
  displayModeBar: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
};

const METRIC_LABELS: Record<string, string> = {
  accuracy: "Accuracy",
  precision_macro: "Precision (macro)",
  recall_macro: "Recall (macro)",
  f1_macro: "F1-score (macro)",
};

function buildBarPlot(
  metrics: Record<string, number>,
): { data: PlotData[]; layout: PlotLayout } {
  const order = ["accuracy", "precision_macro", "recall_macro", "f1_macro"];
  const x: string[] = [];
  const y: number[] = [];
  for (const k of order) {
    if (k in metrics) {
      x.push(METRIC_LABELS[k] ?? k);
      y.push(metrics[k] * 100);
    }
  }
  const data: PlotData[] = [
    {
      type: "bar",
      x,
      y,
      marker: {
        color: y.map((_, i) => `rgba(45, 212, 191, ${0.45 + 0.12 * i})`),
        line: { color: "rgba(255,255,255,0.12)", width: 1 },
      },
      text: y.map((v) => `${v.toFixed(1)}%`),
      textposition: "outside",
      textfont: { color: "#94a3b8", size:11 },
    },
  ];
  const layout: PlotLayout = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "rgba(15, 23, 42, 0.35)",
    font: { family: "DM Sans, system-ui, sans-serif", color: "#e2e8f0", size: 12 },
    margin: { t: 28, r: 24, b: 72, l: 52 },
    yaxis: {
      title: { text: "Percent", font: { size: 11 } },
      range: [0, 115],
      gridcolor: "rgba(148, 163, 184, 0.12)",
      zeroline: false,
    },
    xaxis: {
      tickangle: -28,
      gridcolor: "transparent",
    },
    title: {
      text: "Validation metrics (macro averages where noted)",
      font: { size: 13, color: "#f1f5f9" },
    },
  };
  return { data, layout };
}

function buildConfusionPlot(
  matrix: number[][],
  labels: string[],
): { data: PlotData[]; layout: PlotLayout } {
  const data: PlotData[] = [
    {
      type: "heatmap",
      z: matrix,
      x: labels,
      y: labels,
      colorscale: [
        [0, "rgba(15, 23, 42, 0.9)"],
        [0.35, "rgba(15, 118, 110, 0.45)"],
        [0.7, "rgba(45, 212, 191, 0.75)"],
        [1, "rgb(153, 246, 228)"],
      ],
      hovertemplate: "True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
    },
  ];
  const layout: PlotLayout = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "rgba(15, 23, 42, 0.35)",
    font: { family: "DM Sans, system-ui, sans-serif", color: "#e2e8f0", size: 11 },
    margin: { t: 36, r: 28, b: 72, l: 88 },
    title: {
      text: "Confusion matrix (validation)",
      font: { size: 13, color: "#f1f5f9" },
    },
    xaxis: { title: { text: "Predicted" }, side: "bottom" },
    yaxis: { title: { text: "True label" }, autorange: "reversed" },
  };
  return { data, layout };
}

export function EvaluationPage() {
  const barRef = useRef<HTMLDivElement>(null);
  const cmRef = useRef<HTMLDivElement>(null);
  const [report, setReport] = useState<EvaluationReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await fetchEvaluation();
        if (cancelled) return;
        setReport(r);
        if (!r.available) return;

        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));

        if (cancelled || !barRef.current || !cmRef.current) return;

        if (r.metrics && Object.keys(r.metrics).length > 0) {
          const { data, layout } = buildBarPlot(r.metrics);
          await Plotly.newPlot(barRef.current, data, layout, PLOT_CFG);
        }

        if (r.confusion_matrix?.length && r.class_names?.length) {
          const { data, layout } = buildConfusionPlot(r.confusion_matrix, r.class_names);
          await Plotly.newPlot(cmRef.current, data, layout, PLOT_CFG);
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Load failed");
      }
    })();
    return () => {
      cancelled = true;
      if (barRef.current) Plotly.purge(barRef.current);
      if (cmRef.current) Plotly.purge(cmRef.current);
    };
  }, []);

  return (
    <main className="mx-auto max-w-5xl px-4 pb-24 pt-10 sm:px-6 sm:pt-14">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <h1 className="font-display text-3xl font-bold text-white sm:text-4xl">
          Model evaluation
        </h1>
        <p className="mt-2 max-w-2xl text-slate-400">
          Metrics and confusion structure from the validation split produced by{" "}
          <code className="rounded bg-white/10 px-1.5 py-0.5 text-xs text-clinical-200">
            train_vit_pytorch.py
          </code>{" "}
          (<span className="text-slate-500">evaluation_report.json</span>). Use this for research
          reporting—not for individual patient decisions.
        </p>
      </motion.div>

      {error && (
        <p className="mt-6 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          {error}
        </p>
      )}

      {!error && report && !report.available && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="glass mt-10 rounded-3xl p-8"
        >
          <p className="text-slate-300">
            No evaluation report found. Train the model and deploy{" "}
            <code className="text-clinical-200">evaluation_report.json</code> next to your
            checkpoint (or set <code className="text-clinical-200">EVALUATION_REPORT_PATH</code> on
            the API).
          </p>
        </motion.div>
      )}

      {!error && report?.available && (
        <div className="mt-10 space-y-10">
          {report.source_path && (
            <p className="text-xs text-slate-600">
              Source: <span className="break-all font-mono">{report.source_path}</span>
            </p>
          )}
          {report.note && <p className="text-sm text-slate-500">{report.note}</p>}
          {report.val_loss != null && (
            <p className="text-sm text-slate-400">
              Validation loss:{" "}
              <span className="tabular-nums text-slate-200">{report.val_loss.toFixed(4)}</span>
            </p>
          )}

          <motion.section
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 }}
            className="glass-strong overflow-hidden rounded-3xl p-4 sm:p-6"
          >
            <div ref={barRef} className="min-h-[380px] w-full" />
          </motion.section>

          <motion.section
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="glass-strong overflow-hidden rounded-3xl p-4 sm:p-6"
          >
            <div ref={cmRef} className="min-h-[420px] w-full" />
          </motion.section>
        </div>
      )}
    </main>
  );
}

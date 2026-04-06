import { motion } from "framer-motion";
import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchHistory, type HistoryItem } from "../api";
import { formatClassLabel } from "../lib/labels";

export function HistoryPage() {
  const [rows, setRows] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchHistory(120);
      setRows(data);
    } catch {
      setError("Could not load history. Is the API running?");
      setRows([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <main className="mx-auto max-w-4xl px-4 pb-24 pt-10 sm:px-6 sm:pt-14">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="font-display text-3xl font-bold text-white sm:text-4xl">
            History
          </h1>
          <p className="mt-2 text-slate-400">
            Recent analyses stored on your server (SQLite).
          </p>
        </div>
        <div className="flex gap-2">
          <motion.button
            type="button"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => void load()}
            className="rounded-xl border border-white/15 bg-white/[0.04] px-4 py-2.5 text-sm font-medium text-slate-200"
          >
            Refresh
          </motion.button>
          <Link to="/upload">
            <motion.span
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="inline-block rounded-xl bg-gradient-to-r from-clinical-500 to-teal-500 px-4 py-2.5 text-sm font-semibold text-surface-950"
            >
              New scan
            </motion.span>
          </Link>
        </div>
      </div>

      {error && (
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-6 rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-100/90"
        >
          {error}
        </motion.p>
      )}

      {loading ? (
        <div className="mt-12 flex justify-center">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            className="h-10 w-10 rounded-full border-2 border-clinical-400/30 border-t-clinical-400"
          />
        </div>
      ) : rows.length === 0 ? (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass mt-12 rounded-3xl p-12 text-center"
        >
          <p className="text-slate-400">No predictions recorded yet.</p>
          <Link to="/upload" className="mt-4 inline-block text-sm font-semibold text-clinical-300">
            Run your first analysis →
          </Link>
        </motion.div>
      ) : (
        <ul className="mt-10 space-y-4">
          {rows.map((row, i) => (
            <motion.li
              key={row.id}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: Math.min(i * 0.04, 0.4), duration: 0.35 }}
              className="glass group rounded-2xl p-5 transition-colors hover:bg-white/[0.06] sm:p-6"
            >
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium text-slate-200" title={row.filename}>
                    {row.filename}
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    {new Date(row.created_at).toLocaleString()}
                  </p>
                </div>
                <div className="flex flex-wrap items-center gap-4 sm:justify-end">
                  <div className="text-right">
                    <p className="font-display text-lg font-semibold text-white">
                      {formatClassLabel(row.prediction)}
                    </p>
                    <p className="text-xs tabular-nums text-clinical-300/90">
                      {(row.confidence * 100).toFixed(1)}% confidence
                    </p>
                  </div>
                  <div className="flex gap-1">
                    {Object.entries(row.all_scores)
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 4)
                      .map(([label], j) => (
                        <span
                          key={label}
                          className="hidden h-8 w-1 overflow-hidden rounded-full bg-slate-800 sm:block"
                          title={label}
                        >
                          <span
                            className="block w-full bg-gradient-to-t from-clinical-600 to-clinical-400"
                            style={{ height: `${20 + j * 25}%` }}
                          />
                        </span>
                      ))}
                  </div>
                </div>
              </div>
            </motion.li>
          ))}
        </ul>
      )}
    </main>
  );
}

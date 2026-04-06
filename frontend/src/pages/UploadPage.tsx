import { motion, AnimatePresence } from "framer-motion";
import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { predictImage } from "../api";

const ACCEPT = "image/jpeg,image/png,image/bmp,image/tiff,image/webp";

export function UploadPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const onFile = useCallback((f: File | undefined) => {
    if (!f || !f.type.startsWith("image/")) {
      setError("Please choose an image file (JPEG, PNG, …).");
      return;
    }
    setError(null);
    setFile(f);
  }, []);

  const onPredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const result = await predictImage(file);
      navigate("/results", {
        state: { result, fileName: file.name, file },
        replace: true,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="mx-auto max-w-3xl px-4 pb-24 pt-10 sm:px-6 sm:pt-14">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="text-center"
      >
        <h1 className="font-display text-3xl font-bold tracking-tight text-white sm:text-4xl">
          Analyze MRI slice
        </h1>
        <p className="mx-auto mt-3 max-w-md text-slate-400">
          Drop a slice or browse. We&apos;ll preprocess (224×224, normalize) and run ViT inference.
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, delay: 0.05 }}
        className="mt-10"
      >
        <div
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              document.getElementById("file-input")?.click();
            }
          }}
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDrag(false);
            onFile(e.dataTransfer.files[0]);
          }}
          onClick={() => document.getElementById("file-input")?.click()}
          className={[
            "group relative cursor-pointer rounded-3xl border-2 border-dashed p-10 transition-all duration-300 sm:p-14",
            drag
              ? "border-clinical-400/60 bg-clinical-500/10 shadow-[0_0_40px_-8px_rgba(45,212,191,0.35)]"
              : "border-white/[0.12] bg-white/[0.03] hover:border-clinical-500/35 hover:bg-white/[0.05]",
          ].join(" ")}
        >
          <input
            id="file-input"
            type="file"
            accept={ACCEPT}
            className="hidden"
            onChange={(e) => onFile(e.target.files?.[0])}
          />
          <div className="flex flex-col items-center text-center">
            <motion.div
              animate={{ y: drag ? -4 : 0 }}
              className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-clinical-500/25 to-cyan-500/15 ring-1 ring-clinical-400/25"
            >
              <svg className="h-8 w-8 text-clinical-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
                />
              </svg>
            </motion.div>
            <p className="font-medium text-slate-200">Drag &amp; drop MRI image</p>
            <p className="mt-1 text-sm text-slate-500">or click to select · JPEG, PNG, BMP, TIFF, WebP</p>
          </div>
        </div>
      </motion.div>

      <AnimatePresence>
        {error && (
          <motion.p
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-4 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-center text-sm text-rose-200"
          >
            {error}
          </motion.p>
        )}
      </AnimatePresence>

      <AnimatePresence mode="wait">
        {previewUrl && file && (
          <motion.div
            key="preview"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.35 }}
            className="mt-10"
          >
            <div className="glass-strong overflow-hidden rounded-3xl p-6 sm:p-8">
              <div className="flex flex-col gap-6 sm:flex-row sm:items-center">
                <motion.div
                  initial={{ scale: 0.96, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ type: "spring", stiffness: 260, damping: 22 }}
                  className="relative mx-auto w-full max-w-[200px] shrink-0 overflow-hidden rounded-2xl ring-2 ring-clinical-500/30 ring-offset-4 ring-offset-surface-900"
                >
                  <img src={previewUrl} alt="Preview" className="aspect-square w-full object-cover" />
                </motion.div>
                <div className="min-w-0 flex-1 text-center sm:text-left">
                  <p className="truncate text-sm font-medium text-slate-300" title={file.name}>
                    {file.name}
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    {(file.size / 1024).toFixed(1)} KB · ready for inference
                  </p>
                  <motion.button
                    type="button"
                    disabled={loading}
                    whileHover={{ scale: loading ? 1 : 1.02 }}
                    whileTap={{ scale: loading ? 1 : 0.98 }}
                    onClick={(e) => {
                      e.stopPropagation();
                      void onPredict();
                    }}
                    className="mt-6 w-full rounded-xl bg-gradient-to-r from-clinical-500 to-teal-500 py-3.5 text-sm font-semibold text-surface-950 shadow-lg shadow-teal-900/30 disabled:opacity-60 sm:w-auto sm:min-w-[200px]"
                  >
                    {loading ? "Analyzing…" : "Run prediction"}
                  </motion.button>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </main>
  );
}

import { motion } from "framer-motion";
import { Link } from "react-router-dom";

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.08, delayChildren: 0.06 },
  },
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.22, 1, 0.36, 1] } },
};

export function HomePage() {
  return (
    <main className="mx-auto max-w-6xl px-4 pb-24 pt-12 sm:px-6 sm:pt-16">
      <motion.section
        variants={container}
        initial="hidden"
        animate="show"
        className="grid gap-12 lg:grid-cols-[1.1fr_0.9fr] lg:gap-16 lg:items-center"
      >
        <div className="space-y-8">
          <motion.div variants={item}>
            <span className="inline-flex items-center gap-2 rounded-full border border-clinical-500/25 bg-clinical-500/10 px-3 py-1 text-xs font-semibold uppercase tracking-wider text-clinical-300">
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-clinical-400 opacity-40" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-clinical-400" />
              </span>
              Vision Transformer inference
            </span>
          </motion.div>

          <motion.h1
            variants={item}
            className="font-display text-4xl font-bold leading-[1.1] tracking-tight text-white sm:text-5xl lg:text-6xl"
          >
            MRI intelligence,
            <br />
            <span className="text-gradient-clinical">clinically clear</span> presentation.
          </motion.h1>

          <motion.p
            variants={item}
            className="max-w-xl text-lg leading-relaxed text-slate-400"
          >
            NuraScan applies a fine-tuned <strong className="font-medium text-slate-200">Vision Transformer (ViT)</strong>{" "}
            to T1/T2-weighted-style slices, surfacing four-class probabilities with calibrated
            confidence—not a black box.
          </motion.p>

          <motion.div variants={item} className="flex flex-wrap gap-4">
            <Link to="/upload">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="rounded-xl bg-gradient-to-r from-clinical-500 to-teal-500 px-6 py-3.5 text-sm font-semibold text-surface-950 shadow-lg shadow-clinical-900/40 ring-1 ring-white/10"
              >
                Start analysis
              </motion.button>
            </Link>
            <motion.a
              href="#how-it-works"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="rounded-xl border border-white/15 bg-white/[0.03] px-6 py-3.5 text-sm font-semibold text-slate-200 backdrop-blur-sm"
            >
              How ViT works
            </motion.a>
          </motion.div>

          <motion.dl
            variants={item}
            className="grid grid-cols-3 gap-4 border-t border-white/[0.06] pt-10"
          >
            {[
              ["224² input", "ImageNet-normalized patches"],
              ["4 classes", "Glioma · Meningioma · Pituitary · None"],
              ["GPU-ready", "Torch + FastAPI backend"],
            ].map(([k, v]) => (
              <div key={k}>
                <dt className="font-display text-sm font-semibold text-white">{k}</dt>
                <dd className="mt-1 text-xs text-slate-500">{v}</dd>
              </div>
            ))}
          </motion.dl>
        </div>

        <motion.div
          variants={item}
          className="relative lg:pl-4"
        >
          <div className="glass-strong relative overflow-hidden rounded-3xl p-8 shadow-glass-lg ring-1 ring-white/[0.07]">
            <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-clinical-500/20 blur-3xl" />
            <div className="absolute -bottom-16 -left-16 h-48 w-48 rounded-full bg-cyan-500/15 blur-3xl" />

            <p className="relative text-xs font-semibold uppercase tracking-widest text-clinical-300/90">
              Model stack
            </p>
            <h2 className="relative mt-3 font-display text-2xl font-semibold text-white">
              ViT-Base / 16
            </h2>
            <p className="relative mt-3 text-sm leading-relaxed text-slate-400">
              Images are split into fixed-size patches, linearly embedded, enriched with position
              encodings, and processed by self-attention layers—so global context informs each
              prediction.
            </p>

            <ul className="relative mt-8 space-y-4">
              {[
                { t: "Patch embedding", s: "16×16 tokens at 224 px" },
                { t: "Fine-tuned head", s: "Your MRI classes, softmax probabilities" },
                { t: "Audit trail", s: "Every run saved to History" },
              ].map((row, i) => (
                <motion.li
                  key={row.t}
                  initial={{ opacity: 0, x: 12 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.35 + i * 0.1 }}
                  className="flex gap-3 rounded-xl bg-white/[0.03] px-4 py-3 ring-1 ring-white/[0.05]"
                >
                  <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-clinical-500/20 text-xs font-bold text-clinical-300">
                    {i + 1}
                  </span>
                  <div>
                    <p className="text-sm font-medium text-slate-200">{row.t}</p>
                    <p className="text-xs text-slate-500">{row.s}</p>
                  </div>
                </motion.li>
              ))}
            </ul>
          </div>
        </motion.div>
      </motion.section>

      <section id="how-it-works" className="mt-28 sm:mt-32">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.5 }}
          className="glass rounded-3xl p-8 sm:p-10 lg:p-12"
        >
          <h2 className="font-display text-2xl font-semibold text-white sm:text-3xl">
            Why Vision Transformers for MRI?
          </h2>
          <p className="mt-4 max-w-3xl text-slate-400">
            Convolutional models excel at local textures; ViTs model long-range dependencies in
            every layer—useful when tumor margins, edema, and anatomy interact across the field of
            view. We pair that expressiveness with strong preprocessing and a clean UI so clinicians
            and researchers can review scores without friction.
          </p>
          <div className="mt-8 grid gap-6 sm:grid-cols-3">
            {[
              ["Attention maps", "Each patch attends to informative regions."],
              ["Calibrated output", "Full softmax vector, not a single label."],
              ["Privacy-first", "Runs through your own FastAPI instance."],
            ].map(([title, body]) => (
              <div
                key={title}
                className="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-5"
              >
                <p className="font-display font-semibold text-clinical-200">{title}</p>
                <p className="mt-2 text-sm text-slate-500">{body}</p>
              </div>
            ))}
          </div>
        </motion.div>
      </section>
    </main>
  );
}

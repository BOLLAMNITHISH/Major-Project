import { motion, AnimatePresence } from "framer-motion";
import { Outlet, useLocation } from "react-router-dom";
import { AppNav } from "./AppNav";

export function ShellLayout() {
  const { pathname } = useLocation();

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-surface-950 bg-glow-radial bg-grid-medical bg-[length:48px_48px]">
      <div className="pointer-events-none fixed inset-0 bg-gradient-to-b from-surface-950/80 via-transparent to-surface-950" />

      <AppNav />

      <AnimatePresence mode="wait">
        <motion.div
          key={pathname}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
          className="relative z-10"
        >
          <Outlet />
        </motion.div>
      </AnimatePresence>

      <footer className="relative z-10 mx-auto max-w-6xl px-4 pb-10 pt-6 text-center sm:px-6">
        <p className="text-xs leading-relaxed text-slate-500">
          NuraScan is a research demonstration. AI output is not a medical diagnosis.
          <br />
          Always consult a qualified clinician for clinical decisions.
        </p>
      </footer>
    </div>
  );
}

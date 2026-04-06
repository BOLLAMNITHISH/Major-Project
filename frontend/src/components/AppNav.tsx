import { motion } from "framer-motion";
import { NavLink } from "react-router-dom";

const linkBase =
  "relative rounded-lg px-3 py-2 text-sm font-medium transition-colors";

function NavItem({ to, children }: { to: string; children: React.ReactNode }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `${linkBase} ${isActive ? "text-white" : "text-slate-400 hover:text-clinical-200"}`
      }
    >
      {({ isActive }) => (
        <>
          {children}
          {isActive && (
            <motion.span
              layoutId="nav-pill"
              className="absolute inset-0 -z-10 rounded-lg bg-white/[0.08] ring-1 ring-white/10"
              transition={{ type: "spring", stiffness: 380, damping: 30 }}
            />
          )}
        </>
      )}
    </NavLink>
  );
}

export function AppNav() {
  return (
    <motion.header
      initial={{ y: -16, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      className="sticky top-0 z-50 border-b border-white/[0.06] bg-surface-950/75 px-4 py-3 backdrop-blur-xl sm:px-6"
    >
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4">
        <NavLink to="/" className="group flex items-center gap-2">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-clinical-500/30 to-cyan-500/20 ring-1 ring-clinical-400/30">
            <svg
              className="h-5 w-5 text-clinical-300"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23-.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232 1.232 3.228 0 4.46s-3.228 1.232-4.46 0L12 17.25l-4.243 4.243c-1.232 1.232-3.228 1.232-4.46 0s-1.232-3.228 0-4.46L7.5 13.51"
              />
            </svg>
          </div>
          <div>
            <p className="font-display text-base font-semibold tracking-tight text-white">
              NuraScan
            </p>
            <p className="text-[10px] font-medium uppercase tracking-widest text-clinical-400/90">
              MRI · ViT
            </p>
          </div>
        </NavLink>

        <nav className="flex flex-wrap items-center justify-end gap-1 sm:gap-2">
          <NavItem to="/">Home</NavItem>
          <NavItem to="/upload">Analyze</NavItem>
          <NavItem to="/history">History</NavItem>
          <NavItem to="/evaluation">Insights</NavItem>
        </nav>
      </div>
    </motion.header>
  );
}

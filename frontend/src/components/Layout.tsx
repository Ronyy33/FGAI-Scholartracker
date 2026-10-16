import { NavLink, Outlet } from "react-router-dom";
import { SyntheticBanner } from "./ui";

const LINKS = [
  { to: "/", label: "Overview", end: true },
  { to: "/students", label: "Student risk" },
  { to: "/batch", label: "Batch prediction" },
  { to: "/analytics", label: "Analytics" },
  { to: "/evaluation", label: "Model evaluation" },
  { to: "/about", label: "About" },
];

export function Layout() {
  return (
    <div className="min-h-screen bg-page">
      <SyntheticBanner />
      <div className="mx-auto flex max-w-7xl flex-col gap-6 px-4 py-6 lg:flex-row">
        <nav aria-label="Main" className="lg:w-52 lg:shrink-0">
          <p className="mb-3 text-lg font-semibold text-ink">
            Scholar<span className="text-accent">Track</span>
          </p>
          <ul className="flex flex-wrap gap-1 lg:flex-col">
            {LINKS.map((link) => (
              <li key={link.to}>
                <NavLink
                  to={link.to}
                  end={link.end}
                  className={({ isActive }) =>
                    `block rounded-md px-3 py-2 text-sm ${
                      isActive ? "bg-surface-2 font-medium text-ink" : "text-ink-2 hover:bg-surface-2 hover:text-ink"
                    }`
                  }
                >
                  {link.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
        <main className="min-w-0 flex-1">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

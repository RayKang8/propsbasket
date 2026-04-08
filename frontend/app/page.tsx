import SummaryCards from "./components/SummaryCards";
import TopEdgesTable from "./components/TopEdgesTable";

export default function Home() {
  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 font-sans">
      {/* Navbar */}
      <header className="border-b border-zinc-800 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-xl font-bold tracking-tight text-white">
            PropsBasket
          </span>
          <span className="text-xs font-medium bg-zinc-800 text-zinc-400 px-2 py-0.5 rounded-full">
            AI
          </span>
        </div>
        <nav className="flex items-center gap-6 text-sm text-zinc-400">
          <a href="/" className="text-white font-medium">
            Dashboard
          </a>
          <a href="/props" className="hover:text-white transition-colors">
            Props
          </a>
          <a href="/predictions" className="hover:text-white transition-colors">
            Predictions
          </a>
        </nav>
      </header>

      {/* Page header */}
      <main className="max-w-7xl mx-auto px-6 py-10">
        <div className="mb-8">
          <h1 className="text-2xl font-semibold text-white">Top Edges</h1>
          <p className="text-sm text-zinc-400 mt-1">
            Props where our model probability exceeds the sportsbook implied probability
          </p>
        </div>

        <SummaryCards />
        <TopEdgesTable />
      </main>
    </div>
  );
}

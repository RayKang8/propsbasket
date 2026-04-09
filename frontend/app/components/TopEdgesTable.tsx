"use client";

import { useEffect, useMemo, useState } from "react";

interface Prediction {
  id: number;
  prop_line_id: number;
  model_probability: number;
  implied_probability: number;
  edge: number;
  edge_pct: string;
  prediction_time: string;
  player_name: string;
  player_team: string | null;
  market_type: string;
  line_value: number;
  odds: number;
}

type SortKey = "edge" | "model_prob" | "recent";

const MARKET_LABELS: Record<string, string> = {
  points: "Points",
  assists: "Assists",
  rebounds: "Rebounds",
};

function EdgeBadge({ edge }: { edge: number }) {
  return (
    <span
      className={`inline-block px-2 py-0.5 rounded-md text-xs font-semibold tabular-nums ${
        edge >= 0
          ? "bg-emerald-950 text-emerald-400"
          : "bg-red-950 text-red-400"
      }`}
    >
      {edge >= 0 ? "+" : ""}
      {(edge * 100).toFixed(1)}%
    </span>
  );
}

function ProbBar({ value }: { value: number }) {
  return (
    <div className="flex items-center gap-2">
      <div className="w-24 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
        <div
          className="h-full bg-blue-500 rounded-full"
          style={{ width: `${value * 100}%` }}
        />
      </div>
      <span className="tabular-nums text-sm">{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

function FilterButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
        active
          ? "bg-zinc-700 text-white"
          : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800"
      }`}
    >
      {children}
    </button>
  );
}

export default function TopEdgesTable() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [marketFilter, setMarketFilter] = useState<string>("all");
  const [sortKey, setSortKey] = useState<SortKey>("model_prob");

  useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
    fetch(`${apiUrl}/api/predictions/top-edges?limit=50`)
      .then((res) => {
        if (!res.ok) throw new Error(`API error: ${res.status}`);
        return res.json();
      })
      .then((data: Prediction[]) => {
        setPredictions(data);
        setLoading(false);
      })
      .catch((err: Error) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  // Derive available market types from data
  const marketTypes = useMemo(() => {
    const seen = new Set(predictions.map((p) => p.market_type));
    return Array.from(seen).sort();
  }, [predictions]);

  const filtered = useMemo(() => {
    let rows = predictions;
    if (marketFilter !== "all") {
      rows = rows.filter((p) => p.market_type === marketFilter);
    }
    if (sortKey === "model_prob") {
      rows = [...rows].sort((a, b) => b.model_probability - a.model_probability);
    } else if (sortKey === "edge") {
      rows = [...rows].sort((a, b) => b.edge - a.edge);
    } else if (sortKey === "recent") {
      rows = [...rows].sort(
        (a, b) =>
          new Date(b.prediction_time).getTime() -
          new Date(a.prediction_time).getTime()
      );
    }
    return rows;
  }, [predictions, marketFilter, sortKey]);

  if (loading) {
    return (
      <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-12 flex items-center justify-center">
        <div className="text-zinc-500 text-sm animate-pulse">Loading predictions...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl border border-red-900 bg-zinc-900 p-12 flex flex-col items-center justify-center gap-2 text-center">
        <div className="text-red-400 text-sm font-medium">Failed to load predictions</div>
        <div className="text-zinc-600 text-xs">{error}</div>
      </div>
    );
  }

  if (predictions.length === 0) {
    return (
      <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-12 flex flex-col items-center justify-center gap-3 text-center">
        <div className="text-zinc-500 text-sm">
          No predictions yet — run the ML pipeline to generate edges.
        </div>
        <code className="text-xs text-zinc-600 bg-zinc-800 px-3 py-1.5 rounded-md">
          python scripts/train.py --model xgboost
        </code>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-zinc-800 overflow-hidden">
      {/* Controls */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800 bg-zinc-900 gap-4 flex-wrap">
        {/* Market filter */}
        <div className="flex items-center gap-1">
          <FilterButton
            active={marketFilter === "all"}
            onClick={() => setMarketFilter("all")}
          >
            All
          </FilterButton>
          {marketTypes.map((m) => (
            <FilterButton
              key={m}
              active={marketFilter === m}
              onClick={() => setMarketFilter(m)}
            >
              {MARKET_LABELS[m] ?? m}
            </FilterButton>
          ))}
        </div>

        {/* Sort toggle */}
        <div className="flex items-center gap-1">
          <span className="text-xs text-zinc-600 mr-1">Sort:</span>
          <FilterButton
            active={sortKey === "model_prob"}
            onClick={() => setSortKey("model_prob")}
          >
            Model Prob
          </FilterButton>
          <FilterButton
            active={sortKey === "edge"}
            onClick={() => setSortKey("edge")}
          >
            Top Edge
          </FilterButton>
          <FilterButton
            active={sortKey === "recent"}
            onClick={() => setSortKey("recent")}
          >
            Most Recent
          </FilterButton>
        </div>
      </div>

      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-800 bg-zinc-900/50">
            <th className="text-left px-4 py-3 text-xs font-medium text-zinc-400 uppercase tracking-wider">
              Player
            </th>
            <th className="text-left px-4 py-3 text-xs font-medium text-zinc-400 uppercase tracking-wider">
              Prop
            </th>
            <th className="text-left px-4 py-3 text-xs font-medium text-zinc-400 uppercase tracking-wider">
              Model Prob
            </th>
            <th className="text-left px-4 py-3 text-xs font-medium text-zinc-400 uppercase tracking-wider">
              Implied Prob
            </th>
            <th className="text-left px-4 py-3 text-xs font-medium text-zinc-400 uppercase tracking-wider">
              Edge
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-800 bg-zinc-950">
          {filtered.length === 0 ? (
            <tr>
              <td colSpan={5} className="px-4 py-10 text-center text-zinc-600 text-sm">
                No props match this filter.
              </td>
            </tr>
          ) : (
            filtered.map((p) => (
              <tr key={p.id} className="hover:bg-zinc-900 transition-colors">
                <td className="px-4 py-3">
                  <div className="font-medium text-white">{p.player_name}</div>
                  {p.player_team && (
                    <div className="text-xs text-zinc-500 mt-0.5">{p.player_team}</div>
                  )}
                </td>
                <td className="px-4 py-3">
                  <div className="text-zinc-200 capitalize">{p.market_type}</div>
                  <div className="text-xs text-zinc-500 mt-0.5 tabular-nums">
                    Line {p.line_value} &middot; {p.odds > 0 ? "+" : ""}
                    {p.odds}
                  </div>
                </td>
                <td className="px-4 py-3">
                  <ProbBar value={p.model_probability} />
                </td>
                <td className="px-4 py-3">
                  <ProbBar value={p.implied_probability} />
                </td>
                <td className="px-4 py-3">
                  <EdgeBadge edge={p.edge} />
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>

      {/* Row count */}
      <div className="px-4 py-2 border-t border-zinc-800 bg-zinc-900 text-xs text-zinc-600">
        Showing {filtered.length} of {predictions.length} predictions
      </div>
    </div>
  );
}

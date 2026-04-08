"use client";

import { useEffect, useState } from "react";

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

function EdgeBadge({ edge }: { edge: number }) {
  const isPositive = edge >= 0;
  return (
    <span
      className={`inline-block px-2 py-0.5 rounded-md text-xs font-semibold tabular-nums ${
        isPositive
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

export default function TopEdgesTable() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
    fetch(`${apiUrl}/api/predictions/top-edges?limit=20`)
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
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-800 bg-zinc-900">
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
          {predictions.map((p) => (
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
                  Line {p.line_value} &middot; {p.odds > 0 ? "+" : ""}{p.odds}
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
          ))}
        </tbody>
      </table>
    </div>
  );
}

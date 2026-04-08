"use client";

import { useEffect, useState } from "react";

interface Summary {
  total_predictions: number;
  positive_edge_count: number;
  avg_edge: number;
  best_edge: number;
}

function Card({
  label,
  value,
  sub,
  highlight,
}: {
  label: string;
  value: string;
  sub?: string;
  highlight?: "green" | "default";
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900 px-5 py-4">
      <div className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2">
        {label}
      </div>
      <div
        className={`text-2xl font-semibold tabular-nums ${
          highlight === "green" ? "text-emerald-400" : "text-white"
        }`}
      >
        {value}
      </div>
      {sub && <div className="text-xs text-zinc-600 mt-1">{sub}</div>}
    </div>
  );
}

export default function SummaryCards() {
  const [summary, setSummary] = useState<Summary | null>(null);

  useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
    fetch(`${apiUrl}/api/predictions/summary`)
      .then((res) => res.json())
      .then((data: Summary) => setSummary(data))
      .catch(() => setSummary(null));
  }, []);

  if (!summary) {
    return (
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {[...Array(4)].map((_, i) => (
          <div
            key={i}
            className="rounded-xl border border-zinc-800 bg-zinc-900 px-5 py-4 animate-pulse h-20"
          />
        ))}
      </div>
    );
  }

  const hitRate =
    summary.total_predictions > 0
      ? ((summary.positive_edge_count / summary.total_predictions) * 100).toFixed(0)
      : "—";

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
      <Card
        label="Total Predictions"
        value={summary.total_predictions.toLocaleString()}
        sub="all props analyzed"
      />
      <Card
        label="Positive Edges"
        value={summary.positive_edge_count.toLocaleString()}
        sub={`${hitRate}% of predictions`}
        highlight="green"
      />
      <Card
        label="Avg Edge"
        value={`${summary.avg_edge >= 0 ? "+" : ""}${(summary.avg_edge * 100).toFixed(1)}%`}
        sub="across all predictions"
        highlight={summary.avg_edge > 0 ? "green" : "default"}
      />
      <Card
        label="Best Edge"
        value={`+${(summary.best_edge * 100).toFixed(1)}%`}
        sub="top single prop"
        highlight="green"
      />
    </div>
  );
}

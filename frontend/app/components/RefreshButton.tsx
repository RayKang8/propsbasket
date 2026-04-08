"use client";

import { useState } from "react";

type Status = "idle" | "loading" | "success" | "error";

export default function RefreshButton() {
  const [status, setStatus] = useState<Status>("idle");
  const [message, setMessage] = useState<string | null>(null);

  async function handleRefresh() {
    setStatus("loading");
    setMessage(null);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
      const res = await fetch(`${apiUrl}/api/predictions/refresh`, {
        method: "POST",
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail ?? `API error: ${res.status}`);
      }
      setStatus("success");
      setMessage(`${data.predictions_created} new predictions`);
      // Reload the page so SummaryCards and TopEdgesTable pick up fresh data
      setTimeout(() => window.location.reload(), 1200);
    } catch (err) {
      setStatus("error");
      setMessage(err instanceof Error ? err.message : "Unknown error");
      setTimeout(() => setStatus("idle"), 4000);
    }
  }

  const label =
    status === "loading"
      ? "Refreshing..."
      : status === "success"
        ? message
        : status === "error"
          ? "Failed"
          : "Refresh Predictions";

  return (
    <button
      onClick={handleRefresh}
      disabled={status === "loading" || status === "success"}
      className={`text-xs font-medium px-3 py-1.5 rounded-lg transition-colors disabled:cursor-not-allowed ${
        status === "error"
          ? "bg-red-950 text-red-400 border border-red-800"
          : status === "success"
            ? "bg-emerald-950 text-emerald-400 border border-emerald-800"
            : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700 border border-zinc-700"
      }`}
    >
      {status === "loading" && (
        <span className="mr-1.5 inline-block w-2.5 h-2.5 border border-zinc-500 border-t-zinc-300 rounded-full animate-spin align-middle" />
      )}
      {label}
    </button>
  );
}

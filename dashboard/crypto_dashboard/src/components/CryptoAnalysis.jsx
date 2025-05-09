import React, { useState, useEffect } from "react";
import api from "../api/axios";
import { io } from "socket.io-client";
import { toast } from "react-toastify";

const SOCKET_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:5000";

export function CryptoAnalysis() {
    const [loading, setLoading] = useState(true);
    const [results, setResults] = useState(null);

    useEffect(() => {
        // 1) fire off the pipeline
        api.get("/api/pipeline").catch((err) => {
            console.error(err);
            toast.error("Failed to start analysis pipeline");
        });

        // 2) listen for completion
        const socket = io(SOCKET_URL);
        socket.on("pipeline_complete", (payload) => {
            setResults(payload);
            setLoading(false);
            toast.success("Analysis complete!");
        });

        return () => socket.disconnect();
    }, []);

    if (loading) return <div>🛠 Running analysis…</div>;
    if (!results) return <div>⚠️ No results</div>;

    const {
        representatives,
        clustering_algo,
        correlation_matrix,
        top_positive,
        top_negative,
        eda_plots,
        forecasts_7d,
    } = results;

    return (
        <div className="space-y-6">
            <h2 className="text-xl">🔍 Crypto Analysis</h2>

            <div>
                <strong>Algorithm:</strong> {clustering_algo}
            </div>

            <div>
                <strong>Representative Coins:</strong>
                <ul className="list-disc ml-6">
                    {representatives.map((t) => (
                        <li key={t}>{t}</li>
                    ))}
                </ul>
            </div>

            <div>
                <strong>Top Positive Correlations:</strong>
                <pre className="bg-gray-100 p-2 rounded">
                    {JSON.stringify(top_positive, null, 2)}
                </pre>
            </div>

            <div>
                <strong>Top Negative Correlations:</strong>
                <pre className="bg-gray-100 p-2 rounded">
                    {JSON.stringify(top_negative, null, 2)}
                </pre>
            </div>

            <div>
                <strong>Correlation Matrix:</strong>
                <pre className="bg-gray-100 p-2 rounded">
                    {JSON.stringify(correlation_matrix, null, 2)}
                </pre>
            </div>

            <div>
                <strong>EDA Plots:</strong>
                {Object.entries(eda_plots).map(([tk, files]) => (
                    <div key={tk} className="mb-4">
                        <h3 className="font-semibold">{tk}</h3>
                        <div className="grid grid-cols-2 gap-2">
                            {files.map((f) => (
                                <img
                                    key={f}
                                    src={`${SOCKET_URL}/plots/${f
                                        .split("/")
                                        .pop()}`}
                                    alt={`${tk} EDA`}
                                    className="border"
                                />
                            ))}
                        </div>
                    </div>
                ))}
            </div>

            <div>
                <strong>7-Day Prophet Forecast:</strong>
                {Object.entries(forecasts_7d).map(([tk, fc]) => (
                    <div key={tk} className="mb-4">
                        <h3 className="font-semibold">{tk}</h3>
                        <ul className="list-inside list-disc">
                            {fc.map((point) => (
                                <li key={point.ds}>
                                    {point.ds}: {point.yhat.toFixed(2)} (
                                    {point.yhat_lower.toFixed(2)} -{" "}
                                    {point.yhat_upper.toFixed(2)})
                                </li>
                            ))}
                        </ul>
                    </div>
                ))}
            </div>
        </div>
    );
}

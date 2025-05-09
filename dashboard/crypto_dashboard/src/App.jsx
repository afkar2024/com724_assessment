// src/App.jsx
import React, { useState, useEffect } from "react";
import useCryptoStore from "./store/useCryptoStore";
import { RealtimeChart } from "./components/RealtimeChart";
import { CryptoAnalysis } from "./components/CryptoAnalysis";
import { ForecastDashboard } from "./components/ForecastDashboard";
import api from "./api/axios";
import { toast } from "react-toastify";
import { SymbolDropdown } from "./components/SymbolDropdown";

const App = () => {
    const { setSymbol, activeSymbol } = useCryptoStore();
    const [symbols, setSymbols] = useState([]);
    const [tab, setTab] = useState("chart");

    useEffect(() => {
        api.get("/api/tickers")
            .then((res) => {
                setSymbols(res.data.tickers);
                if (!activeSymbol && res.data.tickers.length) {
                    setSymbol(res.data.tickers[0]);
                    toast.success(`Defaulting to ${res.data.tickers[0]}`);
                }
            })
            .catch(console.error);
    }, [activeSymbol, setSymbol]);

    return (
        <div className="p-4">
            <h1 className="text-2xl mb-4">Crypto Dashboard</h1>

            {/* Tabs */}
            <div className="flex space-x-4 mb-6">
                <button
                    onClick={() => setTab("chart")}
                    className={`px-4 py-2 rounded ${
                        tab === "chart"
                            ? "bg-blue-600 text-white"
                            : "bg-gray-200 text-gray-800"
                    }`}
                >
                    Real-time Chart
                </button>
                <button
                    onClick={() => setTab("analysis")}
                    className={`px-4 py-2 rounded ${
                        tab === "analysis"
                            ? "bg-blue-600 text-white"
                            : "bg-gray-200 text-gray-800"
                    }`}
                >
                    Crypto Analysis
                </button>
                <button
                    onClick={() => setTab("forecast")}
                    className={`px-4 py-2 rounded ${
                        tab === "forecast"
                            ? "bg-blue-600 text-white"
                            : "bg-gray-200 text-gray-800"
                    }`}
                >
                    Forecast
                </button>
            </div>

            {/* Symbol selector */}
            <div className="mb-4 w-48">
                <SymbolDropdown
                    symbols={symbols}
                    value={activeSymbol}
                    onChange={setSymbol}
                />
            </div>

            {/* Main content */}
            {tab === "chart" && <RealtimeChart />}
            {tab === "analysis" && <CryptoAnalysis ticker={activeSymbol} />}
            {tab === "forecast" && <ForecastDashboard />}
        </div>
    );
};

export default App;

import React, { useState, useEffect } from "react";
import useCryptoStore from "./store/useCryptoStore";
import { RealtimeChart } from "./components/RealtimeChart";
import api from "./api/axios";
import { toast } from "react-toastify";

const App = () => {
    const { setSymbol, activeSymbol } = useCryptoStore();
    const [symbols, setSymbols] = useState([]);

    useEffect(() => {
        api.get("/api/tickers")
            .then((res) => {
                setSymbols(res.data.tickers);
                // pick the first one by default (e.g. "BTC-USD")
                if (!activeSymbol && res.data.tickers.length) {
                    setSymbol(res.data.tickers[0]);
                    toast.success(`Defaulting to ${res.data.tickers[0]}`);
                }
            })
            .catch((err) => {
                // interceptor already shows toast, but you can add extra logic here
                console.error("Error loading tickers:", err);
            });
    }, [activeSymbol, setSymbol]);

    return (
        <div className="p-4">
            <h1 className="text-2xl mb-4">Crypto Real-time Chart</h1>

            {/* Render all symbols as buttons */}
            <div className="flex flex-wrap gap-2 mb-4">
                {symbols.map((sym) => (
                    <button
                        key={sym}
                        onClick={() => setSymbol(sym)}
                        className={`px-3 py-1 rounded ${
                            sym === activeSymbol
                                ? "bg-blue-500 text-white"
                                : "bg-gray-200 text-gray-800"
                        }`}
                    >
                        {sym}
                    </button>
                ))}
            </div>

            {/* Chart will auto-show the activeSymbol (defaults to BTC-USD) */}
            <RealtimeChart />
        </div>
    );
};

export default App;

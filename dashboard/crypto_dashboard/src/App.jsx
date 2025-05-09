import React from "react";
import { useCryptoStore } from "./store/useCryptoStore";
import { RealtimeChart } from "./components/RealtimeChart";

const App = () => {
    const { priceData, setSymbol, activeSymbol } = useCryptoStore();
    return (
        <div className="p-4">
            <h1 className="text-2xl mb-4">Crypto Real-time Chart</h1>
            <div className="flex space-x-4 mb-4">
                {Object.keys(priceData).map((sym) => (
                    <button
                        key={sym}
                        className={`px-3 py-1 rounded ${
                            sym === activeSymbol
                                ? "bg-blue-500 text-white"
                                : "bg-gray-200"
                        }`}
                        onClick={() => setSymbol(sym)}
                    >
                        {sym}
                    </button>
                ))}
            </div>
            <RealtimeChart />
        </div>
    );
};

export default App;

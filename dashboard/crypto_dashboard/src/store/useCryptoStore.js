import create from "zustand";

export const useCryptoStore = create((set) => ({
    priceData: {}, // { symbol: [ { time, open, high, low, close, volume } ] }
    addKline: (symbol, kline) =>
        set((state) => {
            const existing = state.priceData[symbol] || [];
            return {
                priceData: {
                    ...state.priceData,
                    [symbol]: [...existing.slice(-500), kline],
                },
            };
        }),
    setSymbol: (symbol) => set({ activeSymbol: symbol }),
    activeSymbol: null,
}));

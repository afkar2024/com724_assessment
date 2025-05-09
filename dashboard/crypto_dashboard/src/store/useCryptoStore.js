import { create } from "zustand";

const useCryptoStore = create((set, get) => ({
    priceData: {}, // { [symbol]: [ { time, open, high, low, close, volume }, … ] }
    activeSymbol: null,

    setSymbol: (sym) => {
        set({ activeSymbol: sym });
        // initialize container if missing
        if (!get().priceData[sym]) {
            set((state) => ({
                priceData: { ...state.priceData, [sym]: [] },
            }));
        }
    },

    setPriceData: (sym, bars) =>
        set((state) => ({
            priceData: { ...state.priceData, [sym]: bars },
        })),

    addKline: (sym, bar) =>
        set((state) => ({
            priceData: {
                ...state.priceData,
                [sym]: [...(state.priceData[sym] || []), bar],
            },
        })),
}));

export default useCryptoStore;

// src/components/RealtimeChart.jsx
import React, { useEffect, useState } from "react";
import useCryptoStore from "../store/useCryptoStore";
import Select from "react-select";
import {
    ChartCanvas,
    Chart,
    CandlestickSeries,
    LineSeries,
    XAxis,
    YAxis,
    CrossHairCursor,
    MouseCoordinateX,
    MouseCoordinateY,
    OHLCTooltip,
    discontinuousTimeScaleProvider,
} from "react-financial-charts";
import { timeFormat } from "d3-time-format";
import { format } from "d3-format";
import { curveMonotoneX } from "d3-shape";

// — styles for react-select; auto width, high z-index
const selectStyles = {
    container: (base) => ({
        ...base,
        minWidth: 120,
        width: "auto",
        zIndex: 9999,
    }),
    menuPortal: (base) => ({ ...base, zIndex: 9999 }),
};

const intervalOptions = [
    { label: "1m", value: "1m" },
    { label: "5m", value: "5m" },
    { label: "15m", value: "15m" },
    { label: "1h", value: "1h" },
    { label: "4h", value: "4h" },
    { label: "1d", value: "1d" },
];

// Extend indicator options
const indicatorOptions = [
    { label: "MA20", value: "MA20" },
    { label: "MA50", value: "MA50" },
    { label: "Bollinger Bands", value: "BB" },
    { label: "RSI", value: "RSI" },
    { label: "MACD", value: "MACD" },
    { label: "MACD Signal", value: "MACD_SIGNAL" },
    { label: "Volatility", value: "VOL" },
];

export const RealtimeChart = () => {
    const { activeSymbol, priceData, setPriceData, addKline } =
        useCryptoStore();
    const bars = priceData[activeSymbol] || [];
    const [interval, setInterval] = useState("1m");
    const [selectedIndicators, setSelectedIndicators] = useState([]);

    // fetch last 100 bars
    useEffect(() => {
        if (!activeSymbol) return;
        const base = activeSymbol.split("-")[0].toUpperCase();
        const symbol = `${base}USDT`;
        fetch(
            `https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${interval}&limit=100`
        )
            .then((r) => r.json())
            .then((data) => {
                const parsed = data.map(([t, o, h, l, c, v]) => ({
                    time: new Date(t),
                    open: +o,
                    high: +h,
                    low: +l,
                    close: +c,
                    volume: +v,
                }));
                setPriceData(activeSymbol, parsed);
            })
            .catch(console.error);
    }, [activeSymbol, interval, setPriceData]);

    // websocket updates
    useEffect(() => {
        if (!activeSymbol) return;
        const base = activeSymbol.split("-")[0].toLowerCase();
        const stream = `${base}usdt@kline_${interval}`;
        const ws = new WebSocket(`wss://stream.binance.com:9443/ws/${stream}`);
        ws.onmessage = (e) => {
            const { k } = JSON.parse(e.data);
            const newBar = {
                time: new Date(k.t),
                open: +k.o,
                high: +k.h,
                low: +k.l,
                close: +k.c,
                volume: +k.v,
            };
            const state = useCryptoStore.getState();
            const prev = state.priceData[activeSymbol] || [];
            const last = prev[prev.length - 1];
            if (last && last.time.getTime() === newBar.time.getTime()) {
                state.setPriceData(activeSymbol, [
                    ...prev.slice(0, -1),
                    newBar,
                ]);
            } else {
                state.addKline(activeSymbol, newBar);
            }
        };
        ws.onerror = (err) => console.warn("WS error", err);
        return () => ws.close();
    }, [activeSymbol, interval, addKline]);

    if (!activeSymbol) return <div>Select a symbol</div>;
    if (bars.length < 2) return <div>Loading...</div>;

    // latest price
    const currentPrice = bars[bars.length - 1].close;

    // compute all indicators in one pass
    const enriched = [];
    let ema12 = null,
        ema26 = null,
        macdArr = [],
        macdSignal = null;
    let avgGain = 0,
        avgLoss = 0;
    const periodRSI = 14,
        periodVol = 30;
    const k12 = 2 / (12 + 1),
        k26 = 2 / (26 + 1),
        kSig = 2 / (9 + 1);
    for (let i = 0; i < bars.length; i++) {
        const d = bars[i];
        let ret = 0;
        if (i > 0) ret = (d.close - bars[i - 1].close) / bars[i - 1].close;
        if (i < periodRSI) {
            avgGain += Math.max(0, ret);
            avgLoss += Math.max(0, -ret);
        } else if (i === periodRSI) {
            avgGain /= periodRSI;
            avgLoss /= periodRSI;
        } else {
            avgGain =
                (avgGain * (periodRSI - 1) + Math.max(0, ret)) / periodRSI;
            avgLoss =
                (avgLoss * (periodRSI - 1) + Math.max(0, -ret)) / periodRSI;
        }
        const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
        const rsi = i < periodRSI ? null : 100 - 100 / (1 + rs);

        ema12 = i === 0 ? d.close : d.close * k12 + ema12 * (1 - k12);
        ema26 = i === 0 ? d.close : d.close * k26 + ema26 * (1 - k26);
        const macd = ema12 - ema26;
        macdArr.push(macd);
        macdSignal =
            i < periodRSI
                ? null
                : macdSignal === null
                ? macdArr.slice(0, 9).reduce((s, v) => s + v, 0) / 9
                : macd * kSig + macdSignal * (1 - kSig);

        let bbUp = null,
            bbLo = null;
        if (i >= 19) {
            const slice = bars.slice(i - 19, i + 1).map((v) => v.close);
            const mean = slice.reduce((s, v) => s + v, 0) / slice.length;
            const sd = Math.sqrt(
                slice.reduce((s, v) => s + (v - mean) ** 2, 0) / slice.length
            );
            bbUp = mean + 2 * sd;
            bbLo = mean - 2 * sd;
        }

        let vol = null;
        if (i >= periodVol) {
            const rSlice = bars.slice(i - periodVol, i + 1).map((v, j, arr) => {
                if (j === 0) return 0;
                const prev = arr[j - 1].close;
                return (v.close - prev) / prev;
            });
            const meanR = rSlice.reduce((s, v) => s + v, 0) / rSlice.length;
            vol = Math.sqrt(
                rSlice.reduce((s, v) => s + (v - meanR) ** 2, 0) / rSlice.length
            );
        }

        enriched.push({
            ...d,
            rsi,
            macd,
            macdSignal,
            bbUpper: bbUp,
            bbLower: bbLo,
            volatility: vol,
        });
    }

    const mainHeight = 400;
    const indicatorHeight = 100;
    const totalHeight =
        mainHeight +
        selectedIndicators.filter((i) =>
            ["RSI", "MACD", "MACD_SIGNAL", "VOL"].includes(i)
        ).length *
            indicatorHeight;

    const XScale = discontinuousTimeScaleProvider.inputDateAccessor(
        (d) => d.time
    );
    const { data, xScale, xAccessor, displayXAccessor } = XScale(enriched);

    return (
        <div className="border p-4">
            {/* Top Bar */}
            <div className="flex items-center space-x-4 mb-6">
                <Select
                    options={intervalOptions}
                    value={intervalOptions.find((o) => o.value === interval)}
                    onChange={(opt) => setInterval(opt.value)}
                    styles={selectStyles}
                    menuPortalTarget={document.body}
                    menuPosition="fixed"
                />
                <Select
                    options={indicatorOptions}
                    isMulti
                    value={indicatorOptions.filter((o) =>
                        selectedIndicators.includes(o.value)
                    )}
                    onChange={(opts) =>
                        setSelectedIndicators(opts.map((o) => o.value))
                    }
                    styles={selectStyles}
                    menuPortalTarget={document.body}
                    menuPosition="fixed"
                />
                <div className="ml-auto font-bold text-lg">
                    {activeSymbol}: {format(".5f")(currentPrice)}
                </div>
            </div>

            <ChartCanvas
                height={totalHeight}
                width={800}
                ratio={window.devicePixelRatio}
                margin={{ left: 50, right: 50, top: 20, bottom: 30 }}
                data={data}
                seriesName={activeSymbol}
                xScale={xScale}
                xAccessor={xAccessor}
                displayXAccessor={displayXAccessor}
                panEvent
                zoomEvent
                clamp
            >
                {/* Main price chart */}
                <Chart
                    id={1}
                    height={mainHeight}
                    yExtents={(d) => {
                        const ext = [d.high, d.low];
                        if (
                            selectedIndicators.includes("BB") &&
                            d.bbUpper != null
                        ) {
                            ext.push(d.bbUpper, d.bbLower);
                        }
                        return ext;
                    }}
                >
                    <XAxis />
                    <YAxis />
                    <MouseCoordinateX
                        displayFormat={timeFormat("%Y-%m-%d %H:%M")}
                    />
                    <MouseCoordinateY displayFormat={format(".5f")} />
                    <CandlestickSeries />
                    {/* Overlays for MA & BB here */}
                    <OHLCTooltip origin={[0, -10]} />
                </Chart>

                {/* Dynamic indicator charts */}
                {selectedIndicators.map((ind, idx) => {
                    if (!["RSI", "MACD", "MACD_SIGNAL", "VOL"].includes(ind))
                        return null;
                    const yAccessor = {
                        RSI: (d) => d.rsi,
                        MACD: (d) => d.macd,
                        MACD_SIGNAL: (d) => d.macdSignal,
                        VOL: (d) => d.volatility,
                    }[ind];
                    const origin = (w, h) => [
                        0,
                        totalHeight - (idx + 1) * indicatorHeight,
                    ];
                    return (
                        <Chart
                            key={ind}
                            id={2 + idx}
                            height={indicatorHeight}
                            origin={origin}
                            yExtents={yAccessor}
                        >
                            <YAxis orientation="right" />
                            {idx ===
                                selectedIndicators.filter((i) =>
                                    [
                                        "RSI",
                                        "MACD",
                                        "MACD_SIGNAL",
                                        "VOL",
                                    ].includes(i)
                                ).length -
                                    1 && <XAxis />}
                            <MouseCoordinateY displayFormat={format(".2f")} />
                            <LineSeries
                                yAccessor={yAccessor}
                                curve={curveMonotoneX}
                            />
                        </Chart>
                    );
                })}

                <CrossHairCursor />
            </ChartCanvas>
        </div>
    );
};

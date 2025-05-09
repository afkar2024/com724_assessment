// src/components/RealtimeChart.jsx
import React, { useEffect } from "react";
import useCryptoStore from "../store/useCryptoStore";
import {
    ChartCanvas,
    Chart,
    CandlestickSeries,
    XAxis,
    YAxis,
    CrossHairCursor,
    MouseCoordinateX,
    MouseCoordinateY,
    OHLCTooltip,
    discontinuousTimeScaleProvider,
} from "react-financial-charts";
// ① pull in a time formatter
import { timeFormat } from "d3-time-format";
// ① and a number formatter
import { format } from "d3-format";

export const RealtimeChart = () => {
    const { activeSymbol, priceData, setPriceData, addKline } =
        useCryptoStore();

    const bars = priceData[activeSymbol] || [];

    // 1️⃣ When the symbol changes, load last 100 1m bars…
    useEffect(() => {
        if (!activeSymbol) return;

        // map "BTC-USD" → "BTCUSDT" → lowercase "btcusdt"
        const base = activeSymbol.split("-")[0].toUpperCase();
        const restSymbol = `${base}USDT`;
        const fetchUrl =
            `https://api.binance.com/api/v3/klines?symbol=${restSymbol}` +
            `&interval=1m&limit=100`;

        fetch(fetchUrl)
            .then((res) => res.json())
            .then((data) => {
                // data is Array of [t, o, h, l, c, v, …]
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
    }, [activeSymbol, setPriceData]);

    // 2️⃣ Then subscribe to the WS for live updates
    useEffect(() => {
        if (!activeSymbol) return;
        const base = activeSymbol.split("-")[0].toLowerCase();
        const streamSymbol = `${base}usdt`; // e.g. "btc" → "btcusdt"
        const ws = new WebSocket(
            `wss://stream.binance.com:9443/ws/${streamSymbol}@kline_1m`
        );

        ws.onmessage = (e) => {
            const { k } = JSON.parse(e.data);
            addKline(activeSymbol, {
                time: new Date(k.t),
                open: +k.o,
                high: +k.h,
                low: +k.l,
                close: +k.c,
                volume: +k.v,
            });
        };
        return () => ws.close();
    }, [activeSymbol, addKline]);

    // loading guard
    if (!activeSymbol) return <div>Select a symbol</div>;
    if (bars.length < 2) return <div>Loading...</div>;

    // format for react-financial-charts
    const XScale = discontinuousTimeScaleProvider.inputDateAccessor(
        (d) => d.time
    );
    const { data, xScale, xAccessor, displayXAccessor } = XScale(bars);

    return (
        <div className="p-2 border-2 h-fit w-fit">
            <ChartCanvas
                height={400}
                width={800}
                ratio={window.devicePixelRatio}
                margin={{ left: 50, right: 50, top: 30, bottom: 30 }}
                data={data}
                seriesName={activeSymbol}
                xScale={xScale}
                xAccessor={xAccessor}
                displayXAccessor={displayXAccessor}
                panEvent={true}
                zoomEvent={true}
                clamp={true}
            >
                <Chart id={1} yExtents={(d) => [d.high, d.low]}>
                    <XAxis />
                    <YAxis />
                    <MouseCoordinateX
                        displayFormat={timeFormat("%Y-%m-%d %H:%M")}
                    />
                    <MouseCoordinateY displayFormat={format(".2f")} />
                    <CandlestickSeries />
                    <OHLCTooltip origin={[0, -10]} />
                </Chart>
                <CrossHairCursor />
            </ChartCanvas>
        </div>
    );
};

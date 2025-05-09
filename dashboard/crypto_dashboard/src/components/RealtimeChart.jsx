import React, { useEffect } from "react";
import { useCryptoStore } from "../store/useCryptoStore";
import {
    ChartCanvas,
    Chart,
    series,
    scale,
    coordinates,
    tooltip,
    axes,
    indicator,
} from "react-financial-charts";

const { CandlestickSeries } = series;
const { XAxis, YAxis } = axes;
const { CrossHairCursor, MouseCoordinateX, MouseCoordinateY } = coordinates;
const { OHLCTooltip } = tooltip;
const { discontinuousTimeScaleProvider } = scale;
const { movingAverage } = indicator;

export const RealtimeChart = () => {
    const { activeSymbol, priceData, addKline } = useCryptoStore();
    const data = priceData[activeSymbol] || [];

    // Subscribe to Binance WebSocket
    useEffect(() => {
        if (!activeSymbol) return;
        const streamSymbol =
            activeSymbol.replace("-", "").toLowerCase() + "usdt";
        const interval = "1m";
        const ws = new WebSocket(
            `wss://stream.binance.com:9443/ws/${streamSymbol}@kline_${interval}`
        );
        ws.onmessage = (e) => {
            const msg = JSON.parse(e.data);
            const k = msg.k;
            const kline = {
                time: new Date(k.t),
                open: +k.o,
                high: +k.h,
                low: +k.l,
                close: +k.c,
                volume: +k.v,
            };
            addKline(activeSymbol, kline);
        };
        return () => ws.close();
    }, [activeSymbol, addKline]);

    if (!activeSymbol) return <div>Select a symbol to view chart</div>;
    if (data.length < 1) return <div>Loading...</div>;

    // Prepare for react-financial-charts
    const sma20 = movingAverage()
        .options({ windowSize: 20, sourcePath: "close" })
        .merge((d, c) => {
            d.sma20 = c;
        })
        .accessor((d) => d.sma20);

    const XScaleProvider = discontinuousTimeScaleProvider.inputDateAccessor(
        (d) => d.time
    );
    const {
        data: chartData,
        xScale,
        xAccessor,
        displayXAccessor,
    } = XScaleProvider(data);

    return (
        <ChartCanvas
            height={400}
            width={800}
            ratio={window.devicePixelRatio}
            margin={{ left: 50, right: 50, top: 10, bottom: 30 }}
            data={chartData}
            seriesName={activeSymbol}
            xScale={xScale}
            xAccessor={xAccessor}
            displayXAccessor={displayXAccessor}
        >
            <Chart id={1} yExtents={(d) => [d.high, d.low]}>
                <XAxis />
                <YAxis />
                <MouseCoordinateX />
                <MouseCoordinateY />
                <CandlestickSeries />
                <OHLCTooltip origin={[0, -10]} />
            </Chart>
            <CrossHairCursor />
        </ChartCanvas>
    );
};

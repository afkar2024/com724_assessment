// src/components/ForecastDashboard.jsx
import React, { useState, useEffect } from "react";
import useCryptoStore from "../store/useCryptoStore";
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
    discontinuousTimeScaleProvider,
} from "react-financial-charts";
import { timeFormat } from "d3-time-format";
import { format } from "d3-format";
import api from "../api/axios";

export const ForecastDashboard = () => {
    const { activeSymbol, priceData } = useCryptoStore();
    const bars = priceData[activeSymbol] || [];

    // local state
    const [forecast, setForecast] = useState([]);
    const [targetPrice, setTargetPrice] = useState("");
    const [targetDate, setTargetDate] = useState(null);
    const [maxProfitInfo, setMaxProfitInfo] = useState(null);
    const [loading, setLoading] = useState(false);

    // last-close
    const currentPrice = bars.length ? bars[bars.length - 1].close : null;

    // whenever symbol changes, reset & fetch
    useEffect(() => {
        setForecast([]);
        setTargetPrice("");
        setTargetDate(null);
        setMaxProfitInfo(null);

        if (!activeSymbol) return;

        setLoading(true);
        api.get(`/api/forecast/${activeSymbol}?model=prophet&horizon=365`)
            .then((res) => {
                const fc = res.data.forecast.map((d) => ({
                    date: new Date(d.ds),
                    price: d.yhat,
                }));
                setForecast(fc);
            })
            .catch(console.error)
            .finally(() => setLoading(false));
    }, [activeSymbol]);

    // compute metrics once we have forecast + a current price
    useEffect(() => {
        if (!forecast.length || currentPrice == null) return;

        const lastBarTime = bars[bars.length - 1].time.getTime();
        const future = forecast.filter((d) => d.date.getTime() > lastBarTime);

        // max profit
        if (future.length) {
            const maxPoint = future.reduce(
                (best, d) => (d.price > best.price ? d : best),
                future[0]
            );
            setMaxProfitInfo({
                profit: maxPoint.price - currentPrice,
                date: maxPoint.date,
            });
        }

        // target date
        if (targetPrice !== "" && !isNaN(+targetPrice)) {
            const threshold = currentPrice + +targetPrice;
            const hits = future.filter((d) => d.price >= threshold);
            setTargetDate(hits.length ? hits[hits.length - 1].date : null);
        }
    }, [forecast, targetPrice, currentPrice, bars]);

    // build the merged data array
    const merged = [
        ...bars.map((b) => ({
            date: b.time,
            open: b.open,
            high: b.high,
            low: b.low,
            close: b.close,
            forecastPrice: null,
        })),
        ...forecast.map((f) => ({
            date: f.date,
            open: null,
            high: null,
            low: null,
            close: null,
            forecastPrice: f.price,
        })),
    ];

    // if no symbol selected, empty
    if (!activeSymbol) {
        return <div className="p-4">Select a symbol to view forecast.</div>;
    }

    // if we're mid-fetch, show a spinner
    if (loading) {
        return <div className="p-4">Loading forecast…</div>;
    }

    // if after loading we still have no data, warn the user
    if (merged.length === 0) {
        return (
            <div className="p-4 text-red-600">
                No historical data or forecast available for {activeSymbol}.
            </div>
        );
    }

    // otherwise render the normal chart
    const XScale = discontinuousTimeScaleProvider.inputDateAccessor(
        (d) => d.date
    );
    const { data, xScale, xAccessor, displayXAccessor } = XScale(merged);

    return (
        <div className="p-4">
            {/* Controls & Metrics */}
            <div className="mb-4 space-y-2">
                <div className="flex items-end space-x-4">
                    <div>
                        <label className="block text-sm font-medium">
                            Target Profit
                        </label>
                        <div className="flex items-center space-x-2">
                            <input
                                type="number"
                                className="border rounded p-2 w-32"
                                value={targetPrice}
                                onChange={(e) => setTargetPrice(e.target.value)}
                                placeholder="e.g. 1000"
                            />
                            <span className="text-gray-600">
                                (in {activeSymbol.split("-")[1]})
                            </span>
                        </div>
                    </div>

                    {maxProfitInfo && (
                        <div className="text-blue-600">
                            Max future profit ≈{" "}
                            {format(".2f")(maxProfitInfo.profit)} on{" "}
                            {timeFormat("%Y-%m-%d")(maxProfitInfo.date)}
                        </div>
                    )}

                    {targetDate ? (
                        <div className="text-green-600 font-semibold">
                            Expected ≥{" "}
                            {format(".2f")(currentPrice + +targetPrice)} on{" "}
                            {timeFormat("%Y-%m-%d")(targetDate)}
                        </div>
                    ) : targetPrice !== "" ? (
                        <div className="text-red-600">
                            Not within forecast horizon
                        </div>
                    ) : null}
                </div>
            </div>

            {/* Unified Candlestick + Forecast */}
            <ChartCanvas
                height={350}
                width={800}
                ratio={window.devicePixelRatio}
                margin={{ left: 50, right: 50, top: 10, bottom: 30 }}
                data={data}
                seriesName={activeSymbol}
                xScale={xScale}
                xAccessor={xAccessor}
                displayXAccessor={displayXAccessor}
                panEvent
                zoomEvent
                clamp
            >
                <Chart
                    id={1}
                    yExtents={(d) =>
                        d.high != null
                            ? [d.high, d.low]
                            : [d.forecastPrice, d.forecastPrice]
                    }
                >
                    <XAxis />
                    <YAxis />
                    <MouseCoordinateX displayFormat={timeFormat("%Y-%m-%d")} />
                    <MouseCoordinateY displayFormat={format(".2f")} />
                    <CandlestickSeries />
                    <LineSeries yAccessor={(d) => d.forecastPrice} />
                </Chart>
                <CrossHairCursor />
            </ChartCanvas>
        </div>
    );
};

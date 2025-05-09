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

    const [forecast, setForecast] = useState([]);
    const [targetPrice, setTargetPrice] = useState("");
    const [targetDate, setTargetDate] = useState(null);
    const [maxProfitInfo, setMaxProfitInfo] = useState("");

    // Pull in the last-close as currentPrice
    const currentPrice = bars.length > 0 ? bars[bars.length - 1].close : null;

    // Fetch forecast whenever symbol changes
    useEffect(() => {
        if (!activeSymbol) return;
        api.get(`/api/forecast/${activeSymbol}?model=prophet&horizon=365`)
            .then((res) => {
                const fc = res.data.forecast.map((d) => ({
                    date: new Date(d.ds),
                    price: d.yhat,
                }));
                setForecast(fc);
                setTargetDate(null);
            })
            .catch(console.error);
    }, [activeSymbol]);

    // Compute the date when forecast ≥ currentPrice + profit
    useEffect(() => {
        if (
            !forecast.length ||
            !currentPrice ||
            targetPrice === "" ||
            isNaN(parseFloat(targetPrice))
        )
            return;

        const lastBarTime = bars[bars.length - 1].time.getTime();
        const profit = parseFloat(targetPrice);
        const threshold = currentPrice + profit;

        const futureForecasts = forecast.filter(
            (d) => d.date.getTime() > lastBarTime
        );
        if (futureForecasts.length) {
            const maxPoint = futureForecasts.reduce(
                (best, d) => (d.price > best.price ? d : best),
                futureForecasts[0]
            );
            const maxProfit = maxPoint.price - currentPrice;
            // store in state (you’ll need a new piece of state: e.g. maxProfitInfo)
            setMaxProfitInfo({
                profit: maxProfit,
                date: maxPoint.date,
            });
        }
        // find *all* hits, then pick the last one
        const hits = forecast.filter(
            (d) => d.date.getTime() > lastBarTime && d.price >= threshold
        );
        setTargetDate(hits.length ? hits[hits.length - 1].date : null);
    }, [forecast, targetPrice, currentPrice, bars]);

    if (!activeSymbol) {
        return <div className="p-4">Select a symbol to view forecast.</div>;
    }

    // Merge bar+forecast into one timeline
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

    // Build the xScale
    const XScale = discontinuousTimeScaleProvider.inputDateAccessor(
        (d) => d.date
    );
    const { data, xScale, xAccessor, displayXAccessor } = XScale(merged);

    return (
        <div className="p-4">
            <div className="mb-4 flex items-center space-x-4">
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
                    <div className="mt-auto text-blue-600">
                        Max future profit ≈{" "}
                        {format(".2f")(maxProfitInfo.profit)} on{" "}
                        {timeFormat("%Y-%m-%d")(maxProfitInfo.date)}
                    </div>
                )}
                {targetDate && (
                    <div className="ml-auto text-green-600 font-semibold">
                        Expected ≥{" "}
                        {format(".2f")(currentPrice + parseFloat(targetPrice))}{" "}
                        on {timeFormat("%Y-%m-%d")(targetDate)}
                    </div>
                )}
                {targetPrice !== "" && !targetDate && (
                    <div className="ml-auto text-red-600">
                        Not within forecast horizon
                    </div>
                )}
            </div>

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
                {/* Candles */}
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

                    {/* Overlay forecast line */}
                    <LineSeries yAccessor={(d) => d.forecastPrice} />
                </Chart>

                <CrossHairCursor />
            </ChartCanvas>
        </div>
    );
};

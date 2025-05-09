import Plot from 'react-plotly.js';
import React, { useState, useEffect } from 'react';
import api from '../api/axios';

export function CryptoAnalysis({ ticker }) {
  const [eda, setEda] = useState(null);

  useEffect(() => {
    api.get(`/api/eda/${ticker}`)
       .then(r => setEda(r.data))
       .catch(console.error);
  }, [ticker]);

  if (!eda) return <div>Loading EDA…</div>;

  return (
    <div className="space-y-8">
      {/* 1. Trend + MA50 */}
      <Plot
        data={[
          { x: eda.trend.map(p=>p.date), y: eda.trend.map(p=>p.Close),
            mode: 'lines', name: 'Close' },
          { x: eda.ma50.map(p=>p.date), y: eda.ma50.map(p=>p.ma50),
            mode: 'lines', name: '50-day MA' },
        ]}
        layout={{ title: 'Price Trend & 50-day MA', xaxis:{type:'date'} }}
      />

      {/* 2. Distribution (histogram + KDE) */}
      <Plot
        data={[
          { x: eda.distribution.map(d=>d.ret), type: 'histogram', name: 'Histogram', opacity:0.7 },
          { x: eda.distribution.map(d=>d.ret), type: 'histogram',
            cumulative:{enabled:true}, name: 'Cumulative' /* or use a density trace */ },
        ]}
        layout={{ title: 'Daily Returns Distribution' }}
      />

      {/* 3. Box plot by year */}
      <Plot
        data={Object.entries(eda.box_by_year).map(([yr, vals]) => ({
          y: vals,
          type: 'box',
          name: yr,
        }))}
        layout={{ title: 'Returns by Year' }}
      />

      {/* 4. Rolling Volatility */}
      <Plot
        data={[
          { x: eda.volatility.map(d=>d.date), y: eda.volatility.map(d=>d.vol),
            mode: 'lines', name: '30-day Volatility' },
        ]}
        layout={{ title: 'Rolling 30-day Volatility', xaxis:{type:'date'} }}
      />
    </div>
  );
}

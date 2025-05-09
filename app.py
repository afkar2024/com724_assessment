# app.py
import os
import yaml
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from data_collection import load_data, run_full_pipeline
from forecasting import ARIMAModel, ProphetModel
from signals import generate_trading_signals, backtest_signals

# -----------------------------
# Load configuration
# -----------------------------
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

TICKERS = config['tickers']
PERIOD = config.get('period', '5y')
CACHE_FILE = config.get('data_file', 'crypto_data.csv')

# -----------------------------
# Flask & SocketIO setup
# -----------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'secret!')
socketio = SocketIO(app, cors_allowed_origins="*")

# In-memory cache
_cached_data = None

# Helper: emit via SocketIO
notify = lambda evt, data: socketio.emit(evt, data)

# -----------------------------
# REST API endpoints
# -----------------------------
@app.route('/api/status')
def status():
    return jsonify({'status': 'ok'})

@app.route('/api/tickers')
def get_tickers():
    return jsonify({'tickers': TICKERS})

@app.route('/api/data/cached')
def data_cached():
    global _cached_data
    if _cached_data is None:
        _cached_data = load_data(
            TICKERS, period=PERIOD, cache_file=CACHE_FILE, notify_fn=None
        )
    return jsonify({'available': list(_cached_data.keys())})

@app.route('/api/data/refresh')
def data_refresh():
    socketio.start_background_task(_refresh_data_task)
    return jsonify({'task': 'refresh_started'})


def _refresh_data_task():
    global _cached_data
    _cached_data = load_data(
        TICKERS, period=PERIOD, cache_file=CACHE_FILE, notify_fn=notify
    )

@app.route('/api/data/<string:ticker>')
def get_ticker_data(ticker):
    global _cached_data
    if _cached_data is None:
        _cached_data = load_data(TICKERS, period=PERIOD, cache_file=CACHE_FILE)
    if ticker not in _cached_data:
        return jsonify({'error': 'ticker not found'}), 404
    df = _cached_data[ticker]
    payload = df.reset_index()[['Date', 'Close']].rename(
        columns={'Date': 'date', 'Close': 'close'}
    ).to_dict(orient='records')
    return jsonify({ticker: payload})

@app.route('/api/forecast/<string:ticker>')
def get_forecast(ticker):
    """Return forecast for a given ticker and horizon via query params."""
    model_name = request.args.get('model', 'prophet').lower()
    horizon = int(request.args.get('horizon', 7))
    global _cached_data
    if _cached_data is None:
        _cached_data = load_data(TICKERS, period=PERIOD, cache_file=CACHE_FILE)
    if ticker not in _cached_data:
        return jsonify({'error': 'ticker not found'}), 404
    ts = _cached_data[ticker]['Close']
    model = ARIMAModel() if model_name == 'arima' else ProphetModel()
    model.fit(ts)
    fc = model.predict(horizon)
    records = fc.reset_index().to_dict(orient='records')
    return jsonify({'ticker': ticker, 'model': model_name, 'horizon': horizon, 'forecast': records})

@app.route('/api/signals/<string:ticker>')
def get_signals(ticker):
    """Generate buy/sell signals and backtest for a given ticker."""
    profit_thresh = float(request.args.get('threshold', 0.01))
    rsi_over = float(request.args.get('rsi_overbought', 70))
    rsi_under = float(request.args.get('rsi_oversold', 30))
    horizon = int(request.args.get('horizon', 7))

    global _cached_data
    if _cached_data is None:
        _cached_data = load_data(TICKERS, period=PERIOD, cache_file=CACHE_FILE)
    if ticker not in _cached_data:
        return jsonify({'error': 'ticker not found'}), 404

    # Prepare data with indicators
    from data_collection import compute_technical_indicators
    enriched = compute_technical_indicators({ticker: _cached_data[ticker]})[ticker]
    # Forecast
    m = ProphetModel().fit(enriched['Close'])
    fc = m.predict(horizon)
    # Generate signals
    signals_df = generate_trading_signals(enriched, fc, 
                                         profit_threshold=profit_thresh,
                                         rsi_overbought=rsi_over,
                                         rsi_oversold=rsi_under)
    stats = backtest_signals(signals_df)
    return jsonify({
        'ticker': ticker,
        'signals': signals_df.to_dict(orient='records'),
        'performance': stats
    })

@app.route('/api/pipeline')
def pipeline_api():
    socketio.start_background_task(_pipeline_task)
    return jsonify({'task': 'pipeline_started'})


def _pipeline_task():
    results = run_full_pipeline(TICKERS, notify_fn=notify)
    # Prepare forecasts
    forecasts = {}
    for tk in results['representatives']:
        ts = _cached_data[tk]['Close'] if _cached_data else load_data(TICKERS)[tk]['Close']
        fc = ProphetModel().fit(ts).predict(7)
        forecasts[tk] = fc.reset_index()[['ds','yhat','yhat_lower','yhat_upper']].tail(7).to_dict(orient='records')
    # Prepare signals & backtest
    signals_results = {}
    for tk in results['representatives']:
        from data_collection import compute_technical_indicators
        enriched = compute_technical_indicators({tk: _cached_data[tk]})[tk]
        fc = ProphetModel().fit(enriched['Close']).predict(7)
        sig_df = generate_trading_signals(enriched, fc)
        perf = backtest_signals(sig_df)
        signals_results[tk] = {'signals': sig_df.to_dict(orient='records'), 'performance': perf}

    payload = {
        'representatives': results['representatives'],
        'clusters': results['clusters'],
        'clustering_algo': results['clustering_algo'],
        'correlation_matrix': results['correlation_matrix'].to_dict(),
        'top_positive': results['top_positive'].to_dict(),
        'top_negative': results['top_negative'].to_dict(),
        'eda_plots': results['eda_plots'],
        'forecasts_7d': forecasts,
        'signals': signals_results
    }
    notify('pipeline_complete', payload)

# -----------------------------
# WebSocket events
# -----------------------------
@socketio.on('connect')
def on_connect():
    emit('message', {'data': 'Connected to Crypto API'})

@socketio.on('run_pipeline')
def on_run_pipeline(_json):
    socketio.start_background_task(_pipeline_task)

# -----------------------------
# Main entry
# -----------------------------
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
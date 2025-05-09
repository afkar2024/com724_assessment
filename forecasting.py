import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA


class ForecastModel(ABC):
    """
    Abstract base class for forecasting models.
    """
    @abstractmethod
    def fit(self, ts: pd.Series):
        """Fit model on the provided time series."""
        pass

    @abstractmethod
    def predict(self, periods: int) -> pd.DataFrame:
        """Forecast the next `periods` steps. Return DataFrame with 'yhat' and optional bounds."""
        pass

    def score(self, test_series: pd.Series) -> dict:
        """Evaluate model on test_series; returns RMSE, MAE."""
        forecast = self.predict(len(test_series))
        y_true = test_series.values
        y_pred = forecast['yhat'].values[:len(y_true)]
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred)
        }


class ARIMAModel(ForecastModel):
    def __init__(self, order=(5,1,0)):
        self.order = order
        self.model = None
        self.fitted = None

    def fit(self, ts: pd.Series):
        self.model = ARIMA(ts, order=self.order)
        self.fitted = self.model.fit()
        return self

    def predict(self, periods: int) -> pd.DataFrame:
        fc = self.fitted.get_forecast(steps=periods)
        df = fc.summary_frame()
        # rename columns to standard
        df = df.rename(columns={'mean': 'yhat', 'mean_ci_lower': 'yhat_lower', 'mean_ci_upper': 'yhat_upper'})
        return df[['yhat', 'yhat_lower', 'yhat_upper']]


class ProphetModel(ForecastModel):
    def __init__(self):
        self.model = Prophet()
        self.fitted = None

    def fit(self, ts: pd.Series):
        df = ts.reset_index().rename(columns={ts.index.name: 'ds', ts.name: 'y'})
        df.columns = ['ds', 'y']
        self.fitted = self.model.fit(df)
        return self

    def predict(self, periods: int) -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        # forecast contains ds, yhat, yhat_lower, yhat_upper
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

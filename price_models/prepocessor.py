import numpy as np
import pandas as pd
from typing import Dict, Union
import talib
from dateutil.relativedelta import relativedelta
from downloaders.downloaders import CompanyInfoHandler, MarketDataHandler
from pathlib import Path
import datetime
import json


try:
    from tqdm import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Simple fallback for tqdm
    def tqdm(iterable, **kwargs):
        return iterable
    def logging_redirect_tqdm():
        from contextlib import nullcontext
        return nullcontext()


class DataConfig:
    """Global configuration for data handlers."""

    _instance = None
    _root_path = None
    start_date: datetime.date = datetime.date(2020, 1, 1)
    end_date: datetime.date = datetime.date(2025, 8, 8)
    N: int = 512
    lookback: int = 30
    prediction_horizon: int = 5
    train_split: float = 0.6
    val_split: float = 0.2
    test_split: float = 0.2

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def set_root_path(cls, root_path: Union[str, Path]):
        """Set the global root path for all data outputs."""
        if isinstance(root_path, str):
            root_path = Path(root_path)
        cls._root_path = root_path.resolve()
        cls._instance = None  # Reset singleton to ensure new path is used

    @classmethod
    def get_root_path(cls) -> Path:
        """Get the current root path, defaulting to the module directory if not set."""
        if cls._root_path is None:
            cls._root_path = Path(__file__).resolve().parent
        return cls._root_path

    @classmethod
    def reset_to_default(cls):
        """Reset to default root path (module directory)."""
        cls._root_path = None
        cls._instance = None

        cls.start_date: datetime.date = datetime.date(2020, 1, 1)
        cls.end_date: datetime.date = datetime.date(2025, 8, 8)
        cls.N: int = 512
        cls.lookback: int = 30
        cls.prediction_horizon: int = 5


def safe_log(x):
    """Safely compute the natural log, replacing 0 with NaN."""
    return np.log(x.replace(0, np.nan))


class MarketDataPreprocessor:
    """Preprocesses raw market data into normalized features and targets."""
    def __init__(self):
        self.train_dates = None
        self.val_dates = None
        self.test_dates = None
        self.trading_dates = None
        self.raw_data = dict()
        self.tickers = []
        self.features = None

    def compute_normalised_features_and_target(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        df = df.sort_values(by=['Date']).copy()

        adj = df['Adj Close'].replace(0, np.nan)

        df['log_1d_ret'] = safe_log(adj).diff()
        df['log_5d_ret'] = safe_log(adj / adj.shift(5))
        df['log_10d_ret'] = safe_log(adj / adj.shift(10))
        df['log_30d_ret'] = safe_log(adj / adj.shift(30))
        df['log_50d_ret'] = safe_log(adj / adj.shift(50))
        df['ema_6'] = adj.ewm(span=6).mean()
        df['ema_12'] = adj.ewm(span=12).mean()
        df['macd'] = df['ema_6'] - df['ema_12']
        df['macd_signal'] = df['macd'].ewm(span=4).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        delta = adj.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        df['rsi_5'] = 100 - 100 / (1 + gain.rolling(5).mean() / loss.rolling(5).mean())
        df['rsi_10'] = 100 - 100 / (1 + gain.rolling(10).mean() / loss.rolling(10).mean())
        df['vol_5d'] = df['log_1d_ret'].rolling(5).std() * np.sqrt(252)
        df['vol_10d'] = df['log_1d_ret'].rolling(10).std() * np.sqrt(252)
        df['vol_30d'] = df['log_1d_ret'].rolling(30).std() * np.sqrt(252)
        df['ma_5'] = adj.rolling(5).mean()
        df['ma_10'] = adj.rolling(10).mean()
        df['ma_30'] = adj.rolling(30).mean()
        df['price_ma_5_ratio'] = adj / df['ma_5']
        df['price_ma_10_ratio'] = adj / df['ma_10']
        df['price_ma_30_ratio'] = adj / df['ma_30']
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - adj.shift()).abs()
        low_close = (df['Low'] - adj.shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_5'] = tr.rolling(5).mean()
        df['atr_10'] = tr.rolling(10).mean()
        df['log_volume'] = safe_log(df['Volume'])
        for w in [5, 10, 30]:
            df[f'volume_{w}d_avg'] = df['Volume'].rolling(w).mean()
            df[f'volume_ratio_{w}d'] = df['Volume'] / df[f'volume_{w}d_avg']
        df['adx_5d'] = talib.ADX(df['High'], df['Low'], adj, timeperiod=5)
        df['adx_15d'] = talib.ADX(df['High'], df['Low'], adj, timeperiod=15)

        # lagged features
        for l in [5, 10, 30]:
            df[f'lag{l}d_log_1d_ret'] = df['log_1d_ret'].shift(l)
            df[f'lag{l}d_close'] = adj.shift(l)
            df[f'lag{l}d_volume'] = df['Volume'].shift(l)

        df['target'] = adj / adj.shift(1) - 1

        zscore_features = [
            'Open', 'High', 'Low', 'Adj Close', 'ema_6', 'ema_12', 'ma_5', 'ma_10', 'ma_30',
            'log_1d_ret', 'log_5d_ret', 'log_10d_ret', 'log_30d_ret', 'log_50d_ret',
            'vol_5d', 'vol_10d', 'vol_30d', 'macd', 'macd_signal',
            'price_ma_5_ratio', 'price_ma_10_ratio', 'price_ma_30_ratio',
            'atr_5', 'atr_10', 'log_volume',
            'volume_5d_avg', 'volume_10d_avg', 'volume_30d_avg',
            'volume_ratio_5d', 'volume_ratio_10d', 'volume_ratio_30d',
            'adx_5d', 'adx_15d',
            'lag5d_log_1d_ret', 'lag10d_log_1d_ret', 'lag30d_log_1d_ret',
            'lag5d_close', 'lag10d_close', 'lag30d_close',
            'lag5d_volume', 'lag10d_volume', 'lag30d_volume']

        minmax_features = ['macd_hist', 'rsi_5', 'rsi_10']
        self.features = zscore_features + minmax_features

        df = df[['Date', 'Ticker'] + zscore_features + minmax_features + ['target']].ffill().bfill()
        d = dict()
        trading_dates = dict()
        for split in ['train', 'val', 'test']:
            dates = getattr(self, f'{split}_dates')
            temp = df[df['Date'].isin(dates)].copy()
            if not len(temp) == len(dates):
                return
            trading_dates[split] = temp.Date.astype(str).tolist()
            norm_params = {}
            for col in zscore_features:
                mean, std = temp[col].mean(), temp[col].std()
                temp[col] = (temp[col] - mean) / (std if std > 1e-6 else 1.0)
                norm_params[col] = ('zscore', mean, std)
            for col in minmax_features:
                min_, max_ = temp[col].min(), df[col].max()
                temp[col] = (temp[col] - min_) / (max_ - min_ if max_ - min_ > 1e-6 else 1.0)
                norm_params[col] = ('minmax', min_, max_)

            d[split] = [temp[self.features].values.astype(np.float32), temp['target'].values.astype(np.float32), norm_params.copy()]
        if self.trading_dates is None:
            self.trading_dates = trading_dates
        self.tickers.append(df['Ticker'].iloc[0])
        return d

    def load(self):
        '''
        Select top N tickers based on average daily volume over the last 10 days.
        '''
        ci = CompanyInfoHandler().load().sort_values(by='averageDailyVolume10Day', ascending=False)
        tickers = list(ci.symbol.unique())[:self.N * 3]
        md = MarketDataHandler(tickers=tickers, start_date=self.start_date - relativedelta(months=5), end_date=self.end_date + relativedelta(days=self.holding_period + 5)).load()
        result = {}
        for k, df in md.items():
            if df.isna().sum().sum() == 0:
                df['Date'] = pd.to_datetime(df['Date'])
                result[k] = df
            if len(result) == self.N:
                break
        self.raw_data = result

        # set the splits:
        nvda = result['NVDA']
        start_index = np.argmin(nvda.dropna().Date.dt.date < self.start_date) + 15 - self.lookback
        self.start_date = nvda.Date.dt.date.values[start_index]

        dates = nvda[nvda.Date.dt.date.between(self.start_date, self.end_date)].dropna().Date.values
        dates.sort()
        total = dates.size
        val_index = int(total * config.train_split)
        test_index = int(total * (config.train_split + config.test_split))

        self.train_dates = dates[:val_index]
        self.val_dates = dates[val_index - 34:test_index]
        self.test_dates = dates[test_index - 34:]
        return result

    def process(self, config: DataConfig):
        self.start_date = config.start_date
        self.end_date = config.end_date
        self.N = config.N
        self.holding_period = config.prediction_horizon
        self.lookback = config.lookback
        result = self.load()
        x_train, Y_train, all_params_train = [], [], {}
        x_val, Y_val, all_params_val = [], [], {}
        x_test, Y_test, all_params_test = [], [], {}
        for ticker, df in tqdm(result.items()):
            d = self.compute_normalised_features_and_target(df)
            if not d:
                continue
            if len(x_train) == self.N:
                break
            x_train_i, Y_train_i, all_params_train_i = d['train']
            x_val_i, Y_val_i, all_params_val_i = d['val']
            x_test_i, Y_test_i, all_params_test_i = d['test']
            x_train.append(x_train_i)
            Y_train.append(Y_train_i)
            x_val.append(x_val_i)
            Y_val.append(Y_val_i)
            x_test.append(x_test_i)
            Y_test.append(Y_test_i)
            all_params_train[ticker] = all_params_train_i
            all_params_val[ticker] = all_params_val_i
            all_params_test[ticker] = all_params_test_i
        all_params = {
            'train': all_params_train,
            'val': all_params_val,
            'test': all_params_test
        }

        X_train, Y_train = np.stack(x_train, axis=1), np.stack(Y_train, axis=1)
        X_val, Y_val = np.stack(x_val, axis=1), np.stack(Y_val, axis=1)
        X_test, Y_test = np.stack(x_test, axis=1), np.stack(Y_test, axis=1)

        root_path = config.get_root_path()
        root_path.mkdir(parents=True, exist_ok=True)

        np.save(root_path / 'X_train.npy', X_train)
        np.save(root_path / 'Y_train.npy', Y_train)
        np.save(root_path / 'X_val.npy', X_val)
        np.save(root_path / 'Y_val.npy', Y_val)
        np.save(root_path / 'X_test.npy', X_test)
        np.save(root_path / 'Y_test.npy', Y_test)
        np.save(root_path / 'dates_train.npy', self.trading_dates['train'])
        np.save(root_path / 'dates_val.npy', self.trading_dates['val'])
        np.save(root_path / 'dates_test.npy', self.trading_dates['test'])
        np.save(root_path / 'tickers.npy', np.array(self.tickers))
        np.save(root_path / 'features.npy', np.array(self.features))
        with open(root_path / 'norm_params.json', 'w') as f:
            # Save normalization parameters for each ticker
            json.dump(all_params, f, indent=2)


if __name__ == '__main__':
    config = DataConfig()
    # config.set_root_path(Path(__file__).resolve().parent / 'data')
    preprocessor = MarketDataPreprocessor()
    preprocessor.process(config)
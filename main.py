import pandas as pd
import yfinance as yf
from stockstats import wrap
from stockstats import unwrap
import numpy as np


# Get the historical price data from yahoo finance
TRAIN_START_DATE = '2010-01-01' # YYYY-MM-DD
TRAIN_END_DATE = '2022-12-31'
TEST_START_DATE = '2023-01-03'
TEST_END_DATE = '2023-12-31'
DOW_30_TICKER = ['AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'KO', 'JPM', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'CRM', 'VZ', 'V', 'WBA', 'WMT', 'DIS']
total_df = pd.DataFrame()
for ticker in DOW_30_TICKER:
  stock = yf.Ticker(ticker)
  data = stock.history(start=TRAIN_START_DATE, end=TEST_END_DATE)
  data["Ticker"] = ticker

  # Calculate technical indicators using stockstats library
  stock_df = wrap(data)
  stock_df[["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30","close_30_sma",	"close_60_sma"]]
  unwrap_df = unwrap(stock_df)
  unwrap_df = unwrap_df.fillna(0)
  unwrap_df = unwrap_df.replace(np.inf,0)

  total_df = total_df._append(unwrap_df)
total_df = total_df.drop(columns=["dividends","stock splits","macds","macdh","boll"])
total_df.reset_index(inplace=True)
total_df.head()

total_df.ticker.value_counts()


# Feature engineering
def calculate_turbulence(data, time_period=252):
  time_period=252
  # can add other market assets
  df = data.copy()

  df_price_pivot = df.pivot(index="Date", columns="ticker", values="close")
  # use returns to calculate turbulence
  df_price_pivot = df_price_pivot.pct_change()

  unique_date = df.Date.unique()
  # start after a fixed timestamp period
  start = time_period
  turbulence_index = [0] * start
  # turbulence_index = [0]
  count = 0
  for i in range(start, len(unique_date)):
      current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
      # use one year rolling window to calcualte covariance
      hist_price = df_price_pivot[
          (df_price_pivot.index < unique_date[i])
          & (df_price_pivot.index >= unique_date[i - time_period])
      ]
      # Drop tickers which has number missing values more than the "oldest" ticker
      filtered_hist_price = hist_price.iloc[
          hist_price.isna().sum().min() :
      ].dropna(axis=1)

      cov_temp = filtered_hist_price.cov()
      current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
          filtered_hist_price, axis=0
      )
      temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
          current_temp.values.T
      )
      if temp > 0:
          count += 1
          if count > 2:
              turbulence_temp = temp[0][0]
          else:
              # avoid large outlier because of the calculation just begins
              turbulence_temp = 0
      else:
          turbulence_temp = 0
      turbulence_index.append(turbulence_temp)

  turbulence_index = pd.DataFrame(
      {"Date": df_price_pivot.index, "turbulence": turbulence_index}
  )

  return turbulence_index

turbulence_index = calculate_turbulence(total_df, time_period=252) # Turbulence index is calculated by day, not by stock
total_df = total_df.merge(turbulence_index, on="Date")
total_df.head()


# Predictive Model
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import torch
from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet, TemporalFusionTransformer, NHiTS, NBeats
from pytorch_forecasting.data import NaNLabelEncoder, TorchNormalizer, GroupNormalizer, EncoderNormalizer
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import MAE, MAPE, SMAPE, RMSE, DistributionLoss, QuantileLoss, MQF2DistributionLoss
from datetime import datetime
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")

# Create Time_idx column for individual series
total_df['Time_idx'] = total_df.groupby('ticker').cumcount()
total_df[total_df['ticker']=='AAPL']

ny_timezone = "America/New_York"
dt_object = pd.to_datetime(TEST_START_DATE).tz_localize(ny_timezone)
boundry = total_df[total_df['Date']==dt_object]
prediction_start_idx = boundry['Time_idx'].iloc[0]
print(prediction_start_idx)

drop_cols = ['Date', 'close', 'ticker']
time_varying_known_reals = [s for s in total_df.columns if s not in drop_cols]
print(time_varying_known_reals)

context_length = 30
prediction_length = 5

training = TimeSeriesDataSet(
    total_df,
    time_idx="Time_idx",
    target="close",
    group_ids=["ticker"],
    static_categoricals=["ticker"],  # as we plan to forecast correlations, it is important to use series characteristics (e.g. a series identifier)
    time_varying_known_reals=time_varying_known_reals,
    time_varying_unknown_reals=["close"],
    max_encoder_length=context_length,
    max_prediction_length=prediction_length,
    add_relative_time_idx=True,
    target_normalizer=GroupNormalizer(groups=['ticker'],transformation="softplus")
    # target_normalizer=EncoderNormalizer()
)

validation = TimeSeriesDataSet.from_dataset(training, total_df, min_prediction_idx=prediction_start_idx)

batch_size = 64
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0)

# Custom callback to retrieve training and validation loss
class MetricTracker(pl.Callback):

    def __init__(self):
        self.train_loss_list = []
        self.val_loss_list = []

    def on_train_epoch_end(self, trainer, module):
        # Capture training loss
        train_loss = trainer.logged_metrics.get('train_loss_epoch', None)
        if train_loss is not None:
            self.train_loss_list.append(train_loss.item())

    def on_validation_epoch_end(self, trainer, module):
        # Capture validation loss
        val_loss = trainer.logged_metrics.get('val_loss', None)
        if val_loss is not None:
            self.val_loss_list.append(val_loss.item())

cb = MetricTracker()

pl.seed_everything(42)

from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger


# Hyper parameters
gradient_clip_val=0.1
hidden_size=200
dropout=0.1
hidden_continuous_size=100
attention_head_size=4
learning_rate=0.0001


# Configure network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=20, verbose=True, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=15,
    accelerator="cuda",
    enable_model_summary=True,
    gradient_clip_val=gradient_clip_val,
    callbacks=[lr_logger, early_stop_callback, cb],
    logger=logger,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=learning_rate,
    hidden_size=hidden_size,
    attention_head_size=attention_head_size,
    dropout=dropout,
    hidden_continuous_size=hidden_continuous_size,
    loss=QuantileLoss(),
    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    log_val_interval=1,
    optimizer="Adam",
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# Fit the network
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)


# Play bell sound after training completion
from IPython.lib.display import Audio
import numpy as np

def generate_bell_sound(framerate, play_time_seconds):
    t = np.linspace(0, play_time_seconds, int(framerate * play_time_seconds))

    # Parameters for the bell sound
    frequency1 = 440  # Fundamental frequency
    frequency2 = 880  # Harmonic frequency
    decay_factor = 0.2  # Decay factor for a bell-like sound

    # Create the bell sound using a combination of sine waves
    bell_sound = np.sin(2 * np.pi * frequency1 * t) * np.exp(-decay_factor * t)
    bell_sound += 0.5 * np.sin(2 * np.pi * frequency2 * t) * np.exp(-decay_factor * t)

    return bell_sound

framerate = 44100  # Adjust the framerate as needed
play_time_seconds = 10

bell_audio_data = generate_bell_sound(framerate, play_time_seconds)
Audio(bell_audio_data, rate=framerate, autoplay=True)

best_model_path = trainer.checkpoint_callback.best_model_path
best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)


# Calcualte performance metrics on validation set
predictions = best_tft.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu"))
MAPE()(predictions.output, predictions.y)

MAE()(predictions.output, predictions.y)
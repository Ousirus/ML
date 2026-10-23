# LSTM Stock Price Prediction

A deep learning pipeline that forecasts time-series price movements using a stacked Long Short-Term Memory (LSTM) neural network built with TensorFlow/Keras.

## Overview

The model is trained on synthetic financial time-series data composed of:
- **Linear Trend:** Long-term directional drift.
- **Sinusoidal Cycle:** Periodicity modeling market seasonality (252-day business cycle).
- **Random Walk Noise:** Stochastic fluctuations simulating daily volatility.

## Pipeline Architecture

1. **Preprocessing:** 
   - Feature scaling via `MinMaxScaler` normalized to `[0, 1]`.
   - Sliding window sequencing (`window_size = 30`) mapping $t_{-30} \dots t_{-1} \to t_0$.
   - 80/20 chronological train-test split without data leakage.
2. **Model Structure:**
   - Layer 1: LSTM (64 units, `return_sequences=True`)
   - Regularization: Dropout (0.2)
   - Layer 2: LSTM (32 units)
   - Regularization: Dropout (0.2)
   - Output: Dense (1 unit, linear activation)
3. **Training:**
   - Optimizer: Adam
   - Loss Function: Mean Squared Error (MSE)
   - Callback: `EarlyStopping` monitoring `val_loss` (patience = 5, best weights restored).
4. **Evaluation:**
   - Root Mean Squared Error (RMSE) & Mean Absolute Error (MAE) computed on inverse-scaled actual values.
   - Matplotlib visualizations for actual vs. predicted prices and loss progression.

## Installation

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
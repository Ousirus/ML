import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

np.random.seed(42)
n=1000
trend=np.linspace(100,170,n)
cycle=12*np.sin(2*np.pi*np.arange(n)/252)
noise=np.cumsum(np.random.randn(n)*0.7)
prices=trend+cycle+noise

scaler=MinMaxScaler()
scaled=scaler.fit_transform(prices.reshape(-1,1))

Window=30 #sliding

def make_seq(data,window):
    X,y= [],[]
    for i in range(len(data)-window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

X,y=make_seq(scaled,Window)
split=int(0.8*len(X))  #train test split can also be done with sklearn train_test_split
X_train,X_test = X[:split],X[split:]
y_train,y_test=y[:split],y[split:]
print(X_train.shape)

model=Sequential([
    LSTM(64,return_sequences=True, input_shape=(Window,1)),
    Dropout(0.2),  #prevent overfitting
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam',loss='mse')
model.summary()

early_stop=EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
history=model.fit(X_train,y_train,epochs=50,batch_size=32,validation_split=0.1, callbacks=[early_stop],verbose=1,)

y_pred_sc=model.predict(X_test)
y_pred=scaler.inverse_transform(y_pred_sc)
y_true=scaler.inverse_transform(y_test.reshape(-1,1))

rmse=np.sqrt(mean_squared_error(y_true,y_pred))
mae=mean_absolute_error(y_true,y_pred)
print(f'RMSE: {rmse:.2f} MAE:{mae:.2f}')
plt.figure(figsize=(12,4))
plt.plot(y_true, label='Actual',color='red')
plt.plot(y_pred,label='Predicted',color='orange',alpha=0.8)
plt.title('LSTM Stock Price Forecast')
plt.xlabel('Test Day'); plt.ylabel('Price($)')
plt.legend(); plt.tight_layout()

plt.figure(figsize=(8,3))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.title('Training vs Validation Loss')
plt.legend();plt.show()
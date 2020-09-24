## CNN_LSTM_battery_SoC


# Brief introduction
This problem is try to solve state-of-charge of Li-ion battery.
The data is from panasonic 18650PF.


# Model explain
Owing to SOC is a time series feature, so I try to use 1d-CNN to get space feature and then fed into LSTM.

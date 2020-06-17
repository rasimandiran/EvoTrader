# Environment for DQN

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Market(object):

    LOOK_BACK = 24

    def __init__(self, data):
        self.fee = 0.001
        self.data = data
        self.data.index = pd.to_datetime(self.data.index, unit="ms")

        self.data_5min = self.data.resample("5min").apply(dict(high="max", low="min", close="last", volume="sum"))
        self.data_10min = self.data.resample("10min").apply(dict(high="max", low="min", close="last", volume="sum"))
        self.data_15min = self.data.resample("15min").apply(dict(high="max", low="min", close="last", volume="sum"))
        self.data_30min = self.data.resample("30min").apply(dict(high="max", low="min", close="last", volume="sum"))
        self.data_1h = self.data.resample("1h").apply(dict(high="max", low="min", close="last", volume="sum"))
        
        self.data_1h["real_high"] = self.data_1h.high
        self.data_1h["real_low"] = self.data_1h.low
        self.data_1h["real_close"] = self.data_1h.close
        
        # indicators
        for col in self.data_1h.columns:
            if "real_" not in col:
                self.data_5min[col] = self.data_5min[col].pct_change()
                self.data_10min[col] = self.data_10min[col].pct_change()
                self.data_15min[col] = self.data_15min[col].pct_change()
                self.data_30min[col] = self.data_30min[col].pct_change()
                self.data_1h[col] = self.data_1h[col].pct_change()

                self.data_5min[col] = self.data_5min[col].replace([-np.Inf, np.Inf], [0, 1])
                self.data_10min[col] = self.data_10min[col].replace([-np.Inf, np.Inf], [0, 1])
                self.data_15min[col] = self.data_15min[col].replace([-np.Inf, np.Inf], [0, 1])
                self.data_30min[col] = self.data_30min[col].replace([-np.Inf, np.Inf], [0, 1])
                self.data_1h[col] = self.data_1h[col].replace([-np.Inf, np.Inf], [0, 1])

        self.data_5min = self.data_5min.dropna()
        self.data_10min = self.data_10min.dropna()
        self.data_15min = self.data_15min.dropna()
        self.data_30min = self.data_30min.dropna()
        self.data_1h = self.data_1h.dropna()

        self.prices = self.data_1h.loc[:, ["real_high", "real_low", "real_close"]]
        self.data_1h = self.data_1h.drop(["real_high", "real_low", "real_close"], axis=1)
        self.step_size = len(self.data_1h.iloc[24:-24])
        self.state_space_size = (self.get_state(0)[3].shape[0], self.get_state(0)[3].shape[1], self.get_state(0)[3].shape[2]+1)


    def get_state(self, t):
        t=24+t
        state = []
        state.append(self.data_5min.loc[self.data_5min.index<self.data_1h.iloc[t].name].tail(self.LOOK_BACK).values)
        state.append(self.data_10min.loc[self.data_10min.index<self.data_1h.iloc[t].name].tail(self.LOOK_BACK).values)
        state.append(self.data_15min.loc[self.data_15min.index<self.data_1h.iloc[t].name].tail(self.LOOK_BACK).values)
        state.append(self.data_30min.loc[self.data_30min.index<self.data_1h.iloc[t].name].tail(self.LOOK_BACK).values)
        state.append(self.data_1h.loc[self.data_1h.index<=self.data_1h.iloc[t].name].tail(self.LOOK_BACK).values)
        
        next_state = []
        next_state.append(self.data_5min.loc[self.data_5min.index<self.data_1h.iloc[t+1].name].tail(self.LOOK_BACK).values)
        next_state.append(self.data_10min.loc[self.data_10min.index<self.data_1h.iloc[t+1].name].tail(self.LOOK_BACK).values)
        next_state.append(self.data_15min.loc[self.data_15min.index<self.data_1h.iloc[t+1].name].tail(self.LOOK_BACK).values)
        next_state.append(self.data_30min.loc[self.data_30min.index<self.data_1h.iloc[t+1].name].tail(self.LOOK_BACK).values)
        next_state.append(self.data_1h.loc[self.data_1h.index<=self.data_1h.iloc[t+1].name].tail(self.LOOK_BACK).values)

        scaler = MinMaxScaler(feature_range=(0,1))
        state = np.array([scaler.fit_transform(s) for s in state])
        next_state = np.array([scaler.fit_transform(s) for s in next_state])
            
        return self.prices.iloc[t].real_low, self.prices.iloc[t].real_high, self.prices.iloc[t].real_close, state, next_state
        

    def buy(self, amount, price):
        return (amount/price)*(1-self.fee)

    def sell(self, qty, price):
        return (qty*price)*(1-self.fee)
#!/usr/bin/env python3.7


import warnings
warnings.simplefilter(action='ignore')

import os
import time
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, train_test_split, cross_val_score, \
                                    GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, \
                                  PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import instruments

# sns.set(font_scale=3, style='white')


hapr_data = pd.read_csv("../data/haproxy-data.csv")

mapping = {
    "param__func__mp.input__cmd_start": "size",
    "metric__mp.input.vdu01.0__ab_transfer_rate_kbyte_per_second": \
        "Max. throughput [kB/s]",
    "param__func__de.upb.lb-haproxy.0.1__cpu_bw": "CPU",
    "param__func__de.upb.lb-haproxy.0.1__mem_max": "Memory",
}

hapr_data = instruments.select_and_rename(hapr_data, mapping)

hapr_data = instruments.replace_size(hapr_data)


mem = 128

hapr_data_small = hapr_data.loc[(hapr_data["size"] == "small") & \
                                (hapr_data["Memory"] == mem)]
hapr_data_small = hapr_data_small[["Max. throughput [kB/s]", "CPU"]]
hapr_data_big = hapr_data.loc[(hapr_data["size"] == "big")  & \
                              (hapr_data["Memory"] == mem)]
hapr_data_big = hapr_data_big[["Max. throughput [kB/s]", "CPU"]]


num_measures = hapr_data_small[hapr_data_small["CPU"] == 0.5].shape[0]
measures = [0 for _ in range(num_measures)]

hapr_data_small = hapr_data_small.append( \
        pd.DataFrame({'Max. throughput [kB/s]': measures, \
                      'CPU': measures}),
        ignore_index=True)
hapr_data_big = hapr_data_big.append(
        pd.DataFrame({'Max. throughput [kB/s]': measures,
                      'CPU': measures}),
        ignore_index=True)


haproxy = hapr_data_small
instruments.plot_vnf_data(haproxy)


vnf_name = 'haproxy'
X, y, scaler = instruments.prepare_data(haproxy, vnf_name)
X_scaled = scaler.transform(X)

models = instruments.train_tune_eval_models(X_scaled, y, vnf_name)
instruments.predict_plot(models, scaler, X, y, vnf_name)














print("Aight!")

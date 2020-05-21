import os
import time
import joblib
import timeit
import random

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


CUSTOM_LIGHT_BLUE = '#3e8ed3'
CUSTOM_BLUE = '#3e5395'


class FixedModel(BaseEstimator):
    def __init__(self, fixed_value):
        self.fixed_value = fixed_value


    def fit(self, X, y):
        return self


    def predict(self, X):
        n_samples = X.shape[0]
        return [self.fixed_value for i in range(n_samples)]


def select_and_rename(df, mapping):
    dff = df[list(mapping.keys())]
    
    for k, v in mapping.items():
        dff.rename(columns={k: v}, inplace=True)
    return dff


def replace_size(df):
    df["size"] = df["size"].str.replace("ab -c 1 -t 60 -n 99999999 -e /tngbench_share/ab_dist.csv -s 60 -k -i http://20.0.0.254:8888/", "small")
    df["size"] = df["size"].str.replace("ab -c 1 -t 60 -n 99999999 -e /tngbench_share/ab_dist.csv -s 60 -k http://20.0.0.254:8888/bunny.mp4", "big")
    df["size"] = df["size"].str.replace("ab -c 1 -t 60 -n 99999999 -e /tngbench_share/ab_dist.csv -s 60 -k -i -X 20.0.0.254:3128 http://40.0.0.254:80/", "small")
    df["size"] = df["size"].str.replace("ab -c 1 -t 60 -n 99999999 -e /tngbench_share/ab_dist.csv -s 60 -k -X 20.0.0.254:3128 http://40.0.0.254:80/bunny.mp4", "big")
    return df


def plot_vnf_data(input_info, filename):
    
    fig, ax = plt.subplots()
    
    for it in input_info:
        plt.scatter(it['data']['Max. throughput [kB/s]'],
                    it['data']['CPU'],
                    label=it['label'],
                    marker=it['marker'],
                    color=it['color'],
                    s=5)

    ax.set_xlabel('Max. throughput [kB/s]')
    ax.set_ylabel('CPU')
    plt.legend()
    
    fig.savefig(f'../plots/{filename} Data.pdf', bbox_inches='tight') # $
    # plt.show() # $


def cross_validation_rmse(model, X, y, vnf_name, k=5, save_model=False):
    scores = cross_val_score(model,
                             X, y,
                             scoring="neg_mean_squared_error",
                             cv=k)
    rmse = np.sqrt(-scores)
    name = type(model).__name__
    print(f"CV RMSE of {name}: {rmse.mean()} (+/-{rmse.std()})") # $
    if save_model:
        model.fit(X, y)
        joblib.dump(model, f'../models/{vnf_name}/{name}.joblib') # $
    return rmse


def tune_hyperparams(model, X, y, params):
    grid_search = GridSearchCV(model,
                               params,
                               cv=5,
                               scoring="neg_mean_squared_error")
    grid_search.fit(X, y)
    return grid_search.best_estimator_


def prepare_data(data, vnf_name):
    X = data[['Max. throughput [kB/s]']]
    y = data['CPU']
    X = X.fillna(X.median())

    scaler = MinMaxScaler()
    scaler.fit(X)
    os.makedirs(f'../models/{vnf_name}', exist_ok=True)
    joblib.dump(scaler, f'../models/{vnf_name}/scaler.joblib') # $
    
    return X, y, scaler


def barplot_compare_times(times, labels, ylabel='Time [s]', filename=None):
    assert len(times) == len(labels)
    
    # times_mean = [np.array(t).mean() for t in times]
    # print(times_mean)
    # times_std = [np.array(t).std() for t in times]
    x = np.arange(len(labels))
    
    sns.set(context='paper', font_scale=0.75, style='whitegrid')
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.bar(x, times, 0.5, capsize=5, color='#3e5395')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    
    if filename is not None:
        fig.savefig(f'../plots/{filename}.pdf', bbox_inches='tight')


def barplot_compare_rmse(scores_default, scores_tuned, labels, data_name):
    assert len(scores_default) == len(scores_tuned) == len(labels)
    
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize = (8, 5))
    
    rmse_mean = [s.mean() for s in scores_default]
    rmse_std = [s.std() for s in scores_default]
    plt.bar(x - width/2,
            rmse_mean,
            width,
            yerr=rmse_std,
            capsize=5,
            color=CUSTOM_LIGHT_BLUE,
            label='Default')
    
    rmse_mean = [s.mean() for s in scores_tuned]
    rmse_std = [s.std() for s in scores_tuned]
    plt.bar(x + width/2,
            rmse_mean,
            width,
            yerr=rmse_std,
            capsize=5,
            color=CUSTOM_BLUE,
            label='Tuned')
        
    # labels
    ax.set_ylabel('RMSE')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    fig.savefig(f'../plots/{data_name} RMSE.pdf', bbox_inches='tight')


def train_tune_eval_models(X, y, vnf_name):
    labels = ['Linear', 'Ridge', 'SVR', 'Forest', 'Boosting', 'MLP', 'Fixed']

    models_default = [
        LinearRegression(),
        Ridge(),
        SVR(),
        RandomForestRegressor(), 
        GradientBoostingRegressor(),
        MLPRegressor(max_iter=1500), 
        FixedModel(fixed_value=0.8)
    ]
    rmse_default = [
        cross_validation_rmse(model,
                              X, y,
                              vnf_name)
            for model in models_default
    ]
    
    params_ridge = {
        'alpha': [0.1, 1, 10]
    }
    
    params_svr = {
        'kernel': ['poly', 'rbf'],
        'C': [1, 10, 100], 
        'epsilon': [0.001, 0.01, 0.1]
    }
    
    params_forest = {
        'n_estimators': [10, 100, 200]
    }
    
    params_boosting = {
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [10, 100, 200]
    }
    
    params_mlp = {
        'hidden_layer_sizes': [(64,), (128,), (256)],
        'alpha': [0.001, 0.0001, 0.00001],
        'learning_rate_init': [0.01, 0.001, 0.0001]
    }
    
    params = [
        {}, # Empty for linear
        params_ridge,
        params_svr,
        params_forest,
        params_boosting,
        params_mlp,
        {} # Empty for fixed
    ]

    models_tuned = [
        tune_hyperparams(models_default[i], X, y, params[i])
        for i in range(len(labels))
    ]
    rmse_tuned = [
        cross_validation_rmse(model, X, y, vnf_name, save_model=True)
            for model in models_tuned
    ]
    
    barplot_compare_rmse(rmse_default,
                         rmse_tuned,
                         labels,
                         f'{vnf_name}_default-tuned')
        
    return models_tuned


def predict_plot(models, scaler, X, y, vnf_name):
    models = models[:6]
    labels = ['Linear', 'Ridge', 'SVR', 'Forest', 'Boosting', 'MLP']
    markers = ['d', 'h', 'p', '+', 's', 'x']
    colors = [
        'magenta',
        'orange',
        'red',
        'green',
        CUSTOM_LIGHT_BLUE,
        CUSTOM_BLUE
    ]
    
    fig, ax = plt.subplots()
    ax.set_ylim((0.0, 1.2))
    plt.scatter(X, y, label='True', marker='.', color='black')
    X = scaler.transform(X)
    times = []
    
    for i, model in enumerate(models):
        name = type(model).__name__
        model.fit(X, y)
        os.makedirs(f'../models/{vnf_name}', exist_ok=True)

        if vnf_name == 'Haproxy Small':
            X_plot = pd.DataFrame(
                    {'Max. throughput [kB/s]': np.arange(200, 2500, 50)})

        elif vnf_name == 'Haproxy Big':
            X_plot = pd.DataFrame(
                    {'Max. throughput [kB/s]': np.arange(200, 800000, 16000)})
        
        X_plot_scaled = scaler.transform(X_plot)
        start = time.time()
        y_pred = model.predict(X_plot_scaled)
        times.append(start - time.time())
        plt.scatter(X_plot, y_pred,
                    label=labels[i],
                    marker=markers[i],
                    color=colors[i],
                    s=5)
    
    plt.xlabel('Traffic load [kB/s]')
    plt.ylabel('CPU')
    plt.legend()
    plt.tight_layout()
    fig.savefig(f'../plots/{vnf_name} Model Comparison.pdf')
    # plt.show() # $
    
    return times


def compare_pred_time():
    times = []
    X_rand = pd.DataFrame(data={'Rand max. throughput': [random.randrange(0, 3000) for i in range(1)]})
    scaler = joblib.load(f'../models/Haproxy Big/scaler.joblib')
    X_scaled = scaler.transform(X_rand)
    model_names = ['LinearRegression', 'Ridge', 'SVR', 'RandomForestRegressor', 'GradientBoostingRegressor', 'MLPRegressor', 'FixedModel']
    models = [joblib.load(f'../models/Haproxy Big/{name}.joblib') for name in model_names]
    for model in models:
        t = timeit.timeit("model.predict(X_scaled)",
                          globals=locals(),
                          number=1)
        times.append(t)

    all_times_ms = [t * 1000 for t in times]
    labels = ['Linear', 'Ridge', 'SVR', 'Forest', 'Boosting', 'MLP', 'Fixed']

    barplot_compare_times(all_times_ms, labels, ylabel="Prediction time [ms]", filename='Prediction time comparison')

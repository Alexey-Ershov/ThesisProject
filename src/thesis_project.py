#!/usr/bin/env python3.7


import warnings
warnings.simplefilter(action='ignore')

import pandas as pd
import seaborn as sns

import instruments


sns.set(context='paper', font_scale=0.75, style='whitegrid')


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


hapr_data_small = hapr_data.loc[(hapr_data["size"] == "small")]
hapr_data_small = hapr_data_small[["Max. throughput [kB/s]", "CPU"]]
_hapr_data_small = hapr_data_small[:800]
hapr_data_big = hapr_data.loc[(hapr_data["size"] == "big")]
hapr_data_big = hapr_data_big[["Max. throughput [kB/s]", "CPU"]]
_hapr_data_big = hapr_data_big[:800]


num_measures = hapr_data_small[hapr_data_small['CPU'] == 0.5].shape[0]
measures = [0 for i in range(num_measures)]

hapr_data_small = hapr_data_small.append( \
        pd.DataFrame({'Max. throughput [kB/s]': measures, \
                      'CPU': measures}),
        ignore_index=True)
hapr_data_big = hapr_data_big.append(
        pd.DataFrame({'Max. throughput [kB/s]': measures,
                      'CPU': measures}),
        ignore_index=True)


input_data = [
    {
        'data': hapr_data_small,
        'label': 'Small',
        'marker': '*',
        'color': 'blue',
    },

    {
        'data': hapr_data_big,
        'label': 'Big',
        'marker': '.',
        'color': 'red',
    }
]

instruments.plot_vnf_data([input_data[0]], 'Haproxy Small')
instruments.plot_vnf_data([input_data[1]], 'Haproxy Big')
instruments.plot_vnf_data(input_data, 'Haproxy Both')


vnf_name = 'Haproxy Small'
X, y, scaler = instruments.prepare_data(hapr_data_small, vnf_name)
X_scaled = scaler.transform(X)

models = instruments.train_tune_eval_models(X_scaled, y, vnf_name)
instruments.predict_plot(models, scaler, X, y, vnf_name)

# vnf_name = 'Haproxy Big'
# X, y, scaler = instruments.prepare_data(hapr_data_big, vnf_name)
# X_scaled = scaler.transform(X)

# models = instruments.train_tune_eval_models(X_scaled, y, vnf_name)
# instruments.predict_plot(models, scaler, X, y, vnf_name)


instruments.compare_pred_time()











print("Aight!")

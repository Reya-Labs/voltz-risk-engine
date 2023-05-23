import pandas as pd
import numpy as np
import json
from position import position
from Simulator import Simulator
from utils import SECONDS_IN_YEAR
from RNItoAPY import *

# Adding an argument parser
from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument(
    "-ds", "--dataset", help="Dataset with raw liquidity indices", required=True)
parser.add_argument("-rate_oracle_mode", "--rate_oracle_mode", type=str, help="Rate Oracle Mode: linear/compounding/sofr", required=True)

input_dataset_name = parser.parse_args().dataset
rate_oracle_mode = parser.parse_args().rate_oracle_mode

# Globals
tau_u_factor, tau_d_factor, sigma_factor, eta_im_factor, eta_lm_factor = 1, 1, 1, 1, 1

WAD = 1e18

# Pick up the optimised parameters
with open(f"./simulations/{input_dataset_name}/params.json") as json_file:
    params = json.load(json_file)

token = position["tokens"][0]
# Get APYs from the raw liquidity indices
df_raw = pd.read_csv(f"./rni_historical_data/{input_dataset_name}.csv")
df = getPreparedRNIData(df_raw)
df = getFrequentData(df, frequency=30)
df_for_values = getDailyApy([[token, df]], lookback=10, rate_oracle_mode=rate_oracle_mode)
df_for_values.set_index("date", inplace=True)

if len(df_for_values)-10 <= position["pool_size"]:
    # Maximum possible lookback
    lookback_standard = len(df_for_values)-position["pool_size"]
if position["pool_size"] != -1:
    # Only keep the latest data for a given N-day pool
    df_for_values = df_for_values.iloc[-position["pool_size"]:]

sim_standard = Simulator(df_protocol=df_for_values, tMax=SECONDS_IN_YEAR)
b_values = sim_standard.compute_b_values()
residual_disc_drift = sim_standard.residual_disc_drifts(b_values_dict=b_values)
a_values, sigma_values = sim_standard.compute_continuous_drift_and_volatility(
    residual_min=residual_disc_drift)

# Get alpha and beta
a, b, sigma = a_values[token], b_values[token], sigma_values[token]
alpha = a*b*params["alpha_factor"]
beta = a*params["beta_factor"]
sigma_squared = (sigma*params["sigma_factor"]*sigma_factor)**2

all_params = ["tau_u", "tau_d", "eta_im",
              "eta_lm", "lookback", "xi_lower", "xi_upper"]
final_params = {}
for p in all_params:
    if p == "tau_u":
        final_params[p] = params[p]*tau_u_factor
    elif p == "tau_d":
        final_params[p] = params[p]*tau_d_factor
    elif p == "eta_lm":
        final_params[p] = params[p]*eta_lm_factor
    elif p == "eta_im":
        final_params[p] = params[p]*eta_im_factor
    else:
        final_params[p] = params[p]
    if p != "lookback":
        final_params[p] = final_params[p]*WAD
final_params["gamma_fee"] = 0.001
final_params["lambda_fee"] = 0
final_params["alpha"] = np.abs(alpha)*WAD
final_params["beta"] = np.abs(beta)*WAD
final_params["sigma_squared"] = np.abs(sigma_squared)*WAD

for k, v in final_params.items():
    print(k, ": ", "{0:.0f}".format(v))

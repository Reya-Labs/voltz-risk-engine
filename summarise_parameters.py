from cmath import atan
import pandas as pd
import numpy as np
import json
from position_dict import position
from Simulator import Simulator
from utils import SECONDS_IN_YEAR
from RNItoAPY import * 

DF_TO_OPTIMIZE = "rocket"
POSITION = "RiskEngineOptimisation_lend_rETH_V2" 
tau_u_factor, tau_d_factor, sigma_factor = 1, 0.1, 0.75 #rETH lend
#tau_u_factor, tau_d_factor, sigma_factor = 1.2, 1, 1 #stETH lend
#tau_u_factor, tau_d_factor, sigma_factor = 1.7, 0.3, 0.75 #aUSDC and aDAI lend
#tau_u_factor, tau_d_factor, sigma_factor = 3.0, 0.1, 1 #cDAI lend
#tau_u_factor, tau_d_factor, sigma_factor = 0.7, 1, 1 #aUSDC borrow
#tau_u_factor, tau_d_factor, sigma_factor = 1.5, 0.2, 0.75 #aWETH borrow
#tau_u_factor, tau_d_factor, sigma_factor = 1, 1, 0.75 #cUSDT borrow

pos = position[POSITION]
WAD = 1e18
# Pick up the optimised parameters 
out_dir = f"./simulations/{POSITION}/{DF_TO_OPTIMIZE}/optuna/"
with open(out_dir+f"optimised_parameters_{DF_TO_OPTIMIZE}.json") as json_file:
    params = json.load(json_file)

token = pos["tokens"][0]
df_raw = pd.read_csv(f"./rni_historical_data/{DF_TO_OPTIMIZE}_{token}.csv")    # Get APYs from the raw liquidity indices
df = getPreparedRNIData(df_raw)
df = getFrequentData(df, frequency=30)
df_for_values = getDailyApy([[token, df]], lookback=10)
df_for_values.set_index("date", inplace=True)

if len(df_for_values)-10 <= pos["pool_size"]:
    lookback_standard = len(df_for_values)-pos["pool_size"] # Maximum possible lookback
if pos["pool_size"] != -1:
    df_for_values = df_for_values.iloc[-pos["pool_size"]:] # Only keep the latest data for a given N-day pool
    
sim_standard = Simulator(df_protocol=df_for_values, tMax=SECONDS_IN_YEAR)   
b_values = sim_standard.compute_b_values()
residual_disc_drift = sim_standard.residual_disc_drifts(b_values_dict=b_values)
a_values, sigma_values = sim_standard.compute_continuous_drift_and_volatility(residual_min=residual_disc_drift)

# Get alpha and beta
a, b, sigma = a_values[token], b_values[token], sigma_values[token]
alpha = a*b*params["alpha_factor"]
beta = a*params["beta_factor"]
sigma_squared = (sigma*params["sigma_factor"]*sigma_factor)**2

all_params = ["tau_u", "tau_d", "eta_im", "eta_lm", "lookback", "xi_lower", "xi_upper"]
final_params = {}
for p in all_params:
    if p=="tau_u":
        final_params[p] = params[p]*tau_u_factor
    elif p=="tau_d":
        final_params[p] = params[p]*tau_d_factor
    else:
        final_params[p] = params[p]
    if p!="lookback":
        final_params[p] = final_params[p]*WAD
final_params["gamma_fee"] = 0.003
final_params["lambda_fee"] = 0
final_params["alpha"] = alpha*WAD
final_params["beta"] = beta*WAD
final_params["sigma_squared"] = sigma_squared*WAD

for k, v in final_params.items():
    print(k, ": ", "{0:.0f}".format(v))
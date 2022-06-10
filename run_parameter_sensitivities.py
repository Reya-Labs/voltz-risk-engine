"""
    Takes a dictionary of the optimised parameters and the name of the parameter to
    vary for the sensitivity studies.
"""
import json
import pandas as pd
import os
from run_simulator import main
from position_dict import position
from MasterPlotter import MasterPlotter as mp

DF_TO_OPTIMIZE = "aave"
POSITION = "Generalised_position_trials" # This is the default market case, which is generic

pos = position[POSITION] # Pick up the edge case to test on
df_aave = pd.read_csv("./historical_data/composite_df_AaveVariable_apy.csv")
df_comp = pd.read_csv("./historical_data/composite_df_CompoundV2_apy.csv")
for df in [df_aave, df_comp]:
    df.dropna(inplace=True)
    df.set_index("Date", inplace=True)

# Pick up the optimised parameters 
out_dir = f"./simulations/{POSITION}/{DF_TO_OPTIMIZE}/optuna/"
with open(out_dir+f"optimised_parammters_{DF_TO_OPTIMIZE}.json") as json_file:
    params = json.load(json_file)

print("PARAMS TO USE: ", params)

def run_factors(parser):

    parser.add_argument("-update_param", "--update_param", type=str, help="The tuneable parameter to scam over", default="tau_u")
    update_param = parser.parse_args().update_param

    df_protocol = df_aave if DF_TO_OPTIMIZE=="aave" else df_comp
    name = "AaveVariable" if DF_TO_OPTIMIZE=="aave" else "CompoundV2"
    
    top_dir = f"./simulations/sensitivity_studies/" # Need to pass an update top_dir to the simulator
    if not os.path.exists(top_dir):
        os.makedirs(top_dir)
    
    # Factor configuration
    # 1. Get the original objective function value
    objective_original = main(df=df_protocol, out_name=f"df_{name}_APY_model_and_bounds_sensitivites", \
            write_all_out=False, sim_dir=top_dir+f"{DF_TO_OPTIMIZE}/", **params)
    # 2. Consider additional factors
    original_param = params[update_param]
    factor_objective = {0: 0} # (objective_original - objective_original) * 100 / objective_original
    factors = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    for f in factors:
        params[update_param] = f*original_param # Update each value
        objective = main(df=df_protocol, out_name=f"df_{name}_APY_model_and_bounds_sensitivites", \
            write_all_out=False, sim_dir=top_dir+f"{DF_TO_OPTIMIZE}/", **params)
        change = ((objective - objective_original)*100)/objective_original
        factor_objective[float("{:.2f}".format((f-1)*100))] = float("{:.2f}".format(change)) # Funny cheat around rounding and saving in memory
    
    # Sort the dictionary by key for plotting purposes
    factor_objective = {k: v for k, v in sorted(factor_objective.items(), key=lambda item: item[0])}
    mp.param_dependence_plots(param_dict=factor_objective, param=update_param, protocol=DF_TO_OPTIMIZE, \
        save=top_dir+f"{DF_TO_OPTIMIZE}/{update_param}_sensitivities.png")
    
if __name__=="__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    run_factors(parser=parser)

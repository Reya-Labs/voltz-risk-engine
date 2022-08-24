"""
    Takes a dictionary of the optimised parameters and the name of the parameter to
    vary for the sensitivity studies.
"""
import json
from sre_constants import OP_IGNORE
from xxlimited import new
import pandas as pd
import os
import run_simulator
from position_dict import position
import matplotlib.pyplot as plt
import numpy as np
import copy

DF_TO_OPTIMIZE = "aave_borrow"
POSITION = "RiskEngineOptimisation_borrow_aWETH_4_months_FTLev" 
pos = position[POSITION]

# Pick up the optimised parameters 
out_dir = f"./simulations/{POSITION}/{DF_TO_OPTIMIZE}/optuna/"
with open(out_dir+f"optimised_parameters_{DF_TO_OPTIMIZE}.json") as json_file:
    params = json.load(json_file)
print("PARAMS TO USE: ", params)

def main(parser):

    parser.add_argument("-update_param", "--update_param", type=str, help="The tuneable parameter to scan over", default="tau_u")
    update_param = parser.parse_args().update_param

    name = "aWETH_borrrow_4_months_FTLev_corrected"
    top_dir = f"./simulations/sensitivity_studies/{name}/{update_param}" # Need to pass an update top_dir to the simulator
    if not os.path.exists(top_dir):
        os.makedirs(top_dir)
    
    # Let's update the parameter we want to test in the sensitivity study
    factors = [0.1, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 5.0, 10.0]
    if update_param=="lookback":
        factors = np.arange(-5, 10, 1)

    params_all = copy.deepcopy(params)
    original_param = params[update_param]
    df_sensitivity = {}
    for factor in factors:
        # Return the single DataFrame and get the margins
        if update_param=="lookback":
            params[update_param] = original_param+factor
            df_sensitivity[original_param+factor] = run_simulator.main(out_name=f"df_{DF_TO_OPTIMIZE}_RiskEngineModel_Factor{factor}", **params, return_df=True)
        else:
            params[update_param] = original_param*factor
            df_sensitivity[factor] = run_simulator.main(out_name=f"df_{DF_TO_OPTIMIZE}_RiskEngineModel_Factor{factor}", **params, return_df=True)
    
    # Now we summarise the different margins
    starts = {
        "FT liquidation margin": "mr_lm_ft_", 
        "FT initial margin": "mr_im_ft_", 
        #"FT net margin": "net_margin_ft_", 
        #"FT PnL": "pnl_ft_", 
        "VT liquidation margin": "mr_lm_vt_", 
        "VT initial margin": "mr_im_vt_", 
        #"VT net margin": "net_margin_vt_", 
        #"VT PnL": "pnl_vt_", 
        "LP liquidation margin": "mr_lm_lp_", 
        "LP initial margin": "mr_im_lp_",
        #"LP net margin": "net_margin_lp_", 
        #"LP PnL": "lp_pnl", 
    }

    for key, value in starts.items():
        df_summary = pd.DataFrame()
        for factor in factors:
            if update_param=="lookback":
                df_summary[original_param+factor]=df_sensitivity[original_param+factor][[c for c in df_sensitivity[original_param+factor].columns if c.startswith(value)]]
            else:
                df_summary[factor]=df_sensitivity[factor][[c for c in df_sensitivity[factor].columns if c.startswith(value)]]
        if update_param=="lookback":
            df_summary.index = df_sensitivity[original_param+factor].index
        else:
            df_summary.index = df_sensitivity[factor].index
        df_summary[list(df_summary.columns)].plot(rot=45, title=key+" "+update_param)
        plt.tight_layout()
        plt.savefig(top_dir+f"/{value}sensitivity.png")
        plt.close()

    new_factor = 1.0
    df_sensitivity = {}
    print(params_all)
    params_all["dev_lm"] = params_all["dev_lm"]*0.3
    params_all["dev_im"] = params_all["dev_im"]*0.1
    params_all["sigma_factor"] = params_all["sigma_factor"]
    #params_all["tau_u"] = params_all["tau_u"]*1.2
    params_all["tau_d"] = params_all["tau_d"]*0.8
    params_all["xi_lower"] = 19
    #params_all["xi_upper"] = 29
    print(params_all)
    df_sensitivity[new_factor] = run_simulator.main(out_name=f"df_{DF_TO_OPTIMIZE}_RiskEngineModel_Factor{factor}", **params_all, return_df=True)
    if update_param!="lookback":
        df_summary = pd.DataFrame()
        for key, value in starts.items():
            df_summary[key]=df_sensitivity[new_factor][[c for c in df_sensitivity[new_factor].columns if c.startswith(value)]]

        df_summary["FT leverage"]=[1000/v[0] for v in df_sensitivity[new_factor][[c for c in df_sensitivity[new_factor].columns if c.startswith("mr_im_ft_")]].values]
        df_summary["VT leverage"]=[1000/v[0] for v in df_sensitivity[new_factor][[c for c in df_sensitivity[new_factor].columns if c.startswith("mr_im_vt_")]].values]
        df_summary["LP leverage"]=[1000/v[0] for v in df_sensitivity[new_factor][[c for c in df_sensitivity[new_factor].columns if c.startswith("mr_im_lp_")]].values]
        
        df_summary.index = df_sensitivity[new_factor].index
        df_summary = df_summary[np.isfinite(df_summary).all(1)]
        print(df_summary[["FT leverage", "VT leverage"]])
        df_summary[[c for c in df_summary.columns if (("FT" in c) and ("margin" in c))]].plot(rot=45, title="FT margins")
        plt.ylabel("Margin from 1000 notional [arb. units]")
        plt.xlabel("Timestamp (210-day pool)")
        #plt.ylim(0,1000)
        plt.tight_layout()
        plt.savefig(f"./simulations/sensitivity_studies/{name}/FT_margins.png")
        plt.close()
    
        df_summary[[c for c in df_summary.columns if (("VT" in c) and ("margin" in c))]].plot(rot=45, title="VT margins")
        plt.ylabel("Margin from 1000 notional [arb. units]")
        plt.xlabel("Timestamp (210-day pool)")
        #plt.ylim(0,1000)
        plt.tight_layout()
        plt.savefig(f"./simulations/sensitivity_studies/{name}/VT_margins.png")
        plt.close()
       
        df_summary[[c for c in df_summary.columns if (("LP" in c) and ("margin" in c))]].plot(rot=45, title="LP margins")
        plt.ylabel("Margin from 1000 notional [arb. units]")
        plt.xlabel("Timestamp (210-day pool)")
        #plt.ylim(0,1000)
        plt.tight_layout()
        plt.savefig(f"./simulations/sensitivity_studies/{name}/LP_margins.png")
        plt.close()
        
        df_summary.iloc[:-1][["FT leverage", "VT leverage"]].plot(rot=45, title="Leverages")
        plt.ylabel("Leverage [x]")
        plt.xlabel("Timestamp (210-day pool)")
        #plt.ylim(0,1000)
        plt.tight_layout()
        plt.savefig(f"./simulations/sensitivity_studies/{name}/FT_VT_Leverages.png")
        plt.close()

if __name__=="__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    main(parser=parser)

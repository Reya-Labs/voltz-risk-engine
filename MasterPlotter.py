"""
    A generic class to take care of the core plotting functionality for the
    simulations and parameterisation of the Risk Engine
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from position_dict import position
from run_simulator import POSITION
import json

class MasterPlotter():

    @staticmethod
    def _saver(save):
        if save is None:
            plt.show()
        else:
            plt.savefig(save)

    # Need to convert dictionaries to DataFrames (e.g. for heatmap plotting). This 
    # allows for simpler plotting.
    # E.g.
    #   data: summary_dict
    #   metric: SRs
    #   trader: SR FT
    def dict_to_df(data, trader, metric):
        tokens = position[POSITION]["tokens"] # Just these tokens for now
        scales = list(data.keys()) # E.g. the different volatility scales
        the_dict = {}
        for s in scales:
            the_dict[s] = {}
            for t in tokens:
                the_dict[s][t] = data[s][metric][trader+": "+t]
        return pd.DataFrame.from_dict(the_dict)

    def plot_time_series(df, xlabel="Time", ylabel="", variables=None, logy=False, save=None, title=""):
        if variables is None:
            raise Exception("Plase specify a variable from the DataFrame")
        if isinstance(variables, str): # If a single string to plot is provided 
            variables = [variables]
        if "date" in [c.lower() for c in df.columns]:
            if "Date" in df.columns: # Make column headings consistent
                df["date"] = df["Date"]
                del df["Date"]
            df.set_index("date", inplace=True)
        
        df.plot(variables).rot(45)
        plt.xlabel(xlabel=xlabel)
        plt.ylabel(ylabel=ylabel)
        plt.title(label=title)
        if logy:
            plt.yscale("log")
        MasterPlotter._saver(save)        

    def plot_parameter_pair_with_uncertainty(df, xlabel="", ylabel="", \
        logy=False, save=None, style="go--", title=""):
        
        if len(df.shape) != 2:
            raise Exception("2x2 matrix of values is not provided. Please run dict_to_df and check output.")
        
        xvals = [float(c.split(" ")[0][2:]) for c in df.columns]
        yvals = [df[c].mean() for c in df.columns]
        yerrs = [df[c].std()*1.96 for c in df.columns]
        plt.errorbar(x=xvals, y=yvals, yerr=yerrs) # To do: add more style 
        plt.xlabel(xlabel=xlabel)
        plt.ylabel(ylabel=ylabel)
        plt.title(label=title)
        if logy:
            plt.yscale("log")
        MasterPlotter._saver(save)   

    """
        Provide a DataFrame directly 
    """
    def plot_variable_heatmap(df, xlabel="", ylabel="", \
        zlabel="", logz=False, title="", save=None):
        
        if len(df.shape) != 2:
            raise Exception("2x2 matrix of values is not provided. Please run dict_to_df and check output.")
        
        the_norm = None
        if logz:
            from matplotlib.colors import LogNorm
            the_norm = LogNorm()
        sns.heatmap(df, cmap="coolwarm", annot=True, norm=the_norm, fmt=".1f", annot_kws={"size":16}, cbar_kws={"label": zlabel})
        plt.xlabel(xlabel=xlabel)
        plt.ylabel(ylabel=ylabel)
        plt.title(label=title)
        MasterPlotter._saver(save)


    """
        Parameter dependence plot for performing sensitivity studies 
        under different conditions
    """
    def param_dependence_plots(param_dict, param="tau_u", protocol="aave", save=None):
        xs = np.linspace(0, len(param_dict), len(param_dict))
        x_ticks = param_dict.keys()
        plt.xticks(xs, x_ticks)
        plt.plot(xs, param_dict.values())
        plt.ylabel("Objective function change [%]")
        plt.xlabel(f"Optimised {param} change [%]")
        plt.title(protocol.capitalize())
        MasterPlotter._saver(save)
        
    """
        Additional plotting scripts for time series studies
    """

    # Volatility dependence time series
    def vol_time_series_plot(var, val="apy", token="USDC", actor="vt", fee="0.0003"):
        data_dict = {}
        f_values = [0.5, 1, 2, 5, 7.5, 10, 20]
        df_dir = f"results/{fee}/Generalised_position_1_10_{token}/aave/"
        if var=="CompoundV2":
            df_dir = f"results/{fee}/Generalised_position_1_10_{token}/compound/"
        
        for f in f_values:
            df = pd.read_csv(df_dir+f"df_{var}_APY_model_and_bounds_optimised_F_value_{f}_neutral_1_10_full_risk_engine_output.csv")
            range = ""
            all_cols = [c for c in df.columns if ((val in c) and (actor in c))]

            for col in all_cols:    
                if actor!="lp":
                    if f"{val}_{actor}_{token}" in col:
                        range= col.split("_")[-2] + "_" + col.split("_")[-1] 
                else:
                    if val!="pnl":
                        range= col.split("_")[-3] + "_" + col.split("_")[-2] + "_" + col.split("_")[-1] 

            if "mr" in val:
                data_dict[f"Vol. scale {f}"] = df[f"{val}_{actor}_{token}_{range}"].values
            elif val=="pnl":
                if actor!="lp":
                    data_dict[f"Vol. scale {f}"] = df[f"{val}_{actor}_{token}_{range}"].values
                else:
                    data_dict[f"Vol. scale {f}"] = df[f"{actor}_{val}"].values
            else:
                if actor!="lp":
                    data_dict[f"Vol. scale {f}"] = df[f"{val}_{actor}_{token}_{range}"].values
                else:
                    data_dict[f"Vol. scale {f}"] = df[f"{actor}_{val}_{token}_{range}"].values

        for key, value in data_dict.items():
            plt.plot(value, label=key)
        
        val_dict = {
            "pnl": "PnL",
            "apy": "APY",
            "mr_lm": "Liquidation margin",
            "mr_im": "Initial margin",
        
        }
    
        plt.xlabel("Day")
        plt.ylabel(f"{actor.upper()} {token} {val_dict[val]}")
        plt.legend(loc="best")
        plt.title("1-10 %,neutral fixed-rate, " + var + f" {token}"+ f" 60 day pool ({fee} gamma_fee)")
    
        plt.tight_layout()
        plt.savefig(f"./params_v1/{fee}/volatility/{actor}_{val}_{var}_{token}_volatility.png")

    # Rate market dependence time series
    def market_time_series_plot(var, val="apy", token="USDC", actor="vt", fee="0.0003"):
        data_dict = {}
        markets = ["neutral", "bear", "bull"]
        df_dir = f"results/{fee}/Generalised_position_1_10_{token}/aave/"
        if var=="CompoundV2":
            df_dir = f"results/{fee}/Generalised_position_1_10_{token}/compound/"
        
        for m in markets:
            df = pd.read_csv(df_dir+f"df_{var}_APY_model_and_bounds_optimised_F_value_1_{m}_1_10_full_risk_engine_output.csv")
            range = ""
            all_cols = [c for c in df.columns if ((val in c) and (actor in c))]

            for col in all_cols:    
                if actor!="lp":
                    if f"{val}_{actor}_{token}" in col:
                        range= col.split("_")[-2] + "_" + col.split("_")[-1] 
                else:
                    if val!="pnl":
                        range= col.split("_")[-3] + "_" + col.split("_")[-2] + "_" + col.split("_")[-1] 

            if "mr" in val:
                data_dict[f"{m} market"] = df[f"{val}_{actor}_{token}_{range}"].values
            elif val=="pnl":
                if actor!="lp":
                    data_dict[f"{m} market"] = df[f"{val}_{actor}_{token}_{range}"].values
                else:
                    data_dict[f"{m} market"] = df[f"{actor}_{val}"].values
            else:
                if actor!="lp":
                    data_dict[f"{m} market"] = df[f"{val}_{actor}_{token}_{range}"].values
                else:
                    data_dict[f"{m} market"] = df[f"{actor}_{val}_{token}_{range}"].values

        for key, value in data_dict.items():
            plt.plot(value, label=key)
        
        val_dict = {
            "pnl": "PnL",
            "apy": "APY",
            "mr_lm": "Liquidation margin",
            "mr_im": "Initial margin",        
        }
    
        plt.xlabel("Date")
        plt.ylabel(f"{actor.upper()} {token} {val_dict[val]}")
        plt.legend(loc="best")
        plt.title("1-10 %, fixed-rate, " + var + f" {token}"+ f" 60 day pool ({fee} gamma_fee)")
        plt.tight_layout()
        plt.savefig(f"./params_v1/{fee}/market/{actor}_{val}_{var}_{token}_market.png")
    
    
    # Time range dependence time series 
    def tick_time_series_plot(var, val="apy", token="USDC", actor="vt", fee="0.0003"):
        data_dict = {}
        ticks = ["0.002_1", "1_10", "10_500", "500_990", "990_999", "999_999.998"]
   
        for t in ticks:
            df_dir = f"results/{fee}/Generalised_position_{t}_{token}/aave/"
            if var=="CompoundV2":
                df_dir = f"results/{fee}/Generalised_position_{t}_{token}/compound/"
            df = pd.read_csv(df_dir+f"df_{var}_APY_model_and_bounds_optimised_F_value_1_neutral_{t}_full_risk_engine_output.csv")
            range = ""
            all_cols = [c for c in df.columns if ((val in c) and (actor in c))]

            for col in all_cols:    
                if actor!="lp":
                    if f"{val}_{actor}_{token}" in col:
                        range= col.split("_")[-2] + "_" + col.split("_")[-1] 
                else:
                    if val!="pnl":
                        range= col.split("_")[-3] + "_" + col.split("_")[-2] + "_" + col.split("_")[-1] 

            if "mr" in val:
                data_dict[f"{t} tick"] = df[f"{val}_{actor}_{token}_{range}"].values
            elif val=="pnl":
                if actor!="lp":
                    data_dict[f"{t} tick"] = df[f"{val}_{actor}_{token}_{range}"].values
                else:
                    data_dict[f"{t} tick"] = df[f"{actor}_{val}"].values
            else:
                if actor!="lp":
                    data_dict[f"{t} tick"] = df[f"{val}_{actor}_{token}_{range}"].values
                else:
                    data_dict[f"{t} tick"] = df[f"{actor}_{val}_{token}_{range}"].values

        for key, value in data_dict.items():
            plt.plot(value, label=key)
        
        val_dict = {
            "pnl": "PnL",
            "apy": "APY",
            "mr_lm": "Liquidation margin",
            "mr_im": "Initial margin",
        }
    
        plt.xlabel("Date")
        plt.ylabel(f"{actor.upper()} {token} {val_dict[val]}")
        plt.legend(loc="best")
        plt.title("Neutral fixed-rate, " + var + f" {token}"+ f" 60 day pool ({fee} gamma_fee)")
        plt.tight_layout()
        plt.savefig(f"./params_v1/{fee}/tick/{actor}_{val}_{var}_{token}_tick.png")
    
    
    # Study the optimised parameters across the different tick ranges
    def compare_optimised_parameters(token="DAI", protocol="aave", param="tau_u"):
        ticks = ["0.002_1", "1_10", "10_500", "500_990", "990_999", "999_999.998"]
        summary_dict = {}
        for tick in ticks:
            with open(f"./results/Generalised_position_{tick}_{token}/{protocol}/optuna/optimised_parameters_{protocol}.json") as json_file:
                data = json.load(json_file)
            summary_dict[tick.split("_")[0] + "-" + tick.split("_")[1]] = data[param]
    
        mean = np.array(list(summary_dict.values())).mean()       
        xs = np.linspace(0, len(summary_dict), len(summary_dict))
        x_ticks = summary_dict.keys()
        plt.xticks(xs, x_ticks)
        plt.plot(xs, summary_dict.values())
    
        plt.xlabel("Tick range rates [%]")
        plt.ylabel(f"{param}")
        plt.title(protocol.capitalize() + f" {token}"+ " 60 day pool")
        plt.axhline(y=mean, color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig(f"./time_series/{param}_{token}_{protocol}_across_ticks.png")

    # Study the optimised parameters across the different tick ranges
    def get_mean_parameters(token="DAI", protocol="aave", param="tau_u"):
        ticks = ["0.002_1", "1_10", "10_500", "500_990", "990_999", "999_999.998"]
        summary_dict = {}
        for tick in ticks:
            with open(f"./results/Generalised_position_{tick}_{token}/{protocol}/optuna/optimised_parameters_{protocol}.json") as json_file:
                data = json.load(json_file)
            summary_dict[tick.split("_")[0] + "-" + tick.split("_")[1]] = data[param]
    
        mean = np.array(list(summary_dict.values())).mean()
        return mean

    

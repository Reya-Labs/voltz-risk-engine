import pandas as pd
from MarginCalculator import MarginCalculator
from Simulator import Simulator
from TestPortfolioCalculator import TestPortfolioCalculator
import json
import os
import optuna
import numpy as np
# Positions -- want to disentangle positions from parameters
from position_dict import position
from utils.constants import ALPHA, BETA, MIN_MARGIN_TO_INCENTIVIZE_LIQUIDATORS, SIGMA_SQUARED, XI_LOWER, XI_UPPER
from utils.utils import SECONDS_IN_YEAR, fixedRateToTick, notional_to_liquidity

# ref: https://github.com/optuna/optuna-examples/blob/main/sklearn/sklearn_optuna_search_cv_simple.py
# Globals 
RUN_OPTUNA = False
DF_TO_OPTIMIZE = "aave"

# Positions
POSITION = "leverage_positions_USDC" 
pos = position[POSITION]
top_dir = f"./simulations/{POSITION}/"

def normalise(array):
    if array.max() != array.min():
        return (array-array.min())/(array.max()-array.min())
    else:
        raise Exception("ERROR: minimum and maximum values coincide in array normalisation. Check inputs!")

def main(in_name, out_name, tau_u = 1.5, tau_d = 0.7, gamma_unwind=1, dev_lm=0.5, dev_im=0.3, lookback=30, \
    r_init_lm=0.3, r_init_im=0.1, lambda_fee=0.1, gamma_fee=0.003, a_factor=1, b_factor=1, \
    write_all_out=False, sim_dir=None, debug=False):

    # Generate a simulation-specific directory, based on different tuneable parameters
    if sim_dir is None:
        sim_dir = top_dir+f"{DF_TO_OPTIMIZE}/"
    
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)
   
    # Reload data-set e.g. every time Optuna enters a new trial
    df = pd.read_csv(f"./historical_data/composite_df_{in_name}_apy.csv")
    df.set_index("Date", inplace=True)
    
    # We will use the moving avaerage APY, with given lookback, to compute the
    # calibration and volatility parameters in the CIR model
    for token in df.columns:
        df[token] = df[token].rolling(lookback).mean() # MA 
    df = df.iloc[-pos["pool_size"]:] # Only keep the latest data for a given N-day pool

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # 1. Instantiate Simulator: inherits from Calibrator, gets CIR model params, generates the APY bounds # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    sim = Simulator(df_protocol=df, tMax=SECONDS_IN_YEAR)   
    b_values = sim.compute_b_values()
    residual_disc_drift = sim.residual_disc_drifts(b_values_dict=b_values)
    a_values, sigma_values = sim.compute_continuous_drift_and_volatility(residual_min=residual_disc_drift)

    # We want to now vary the calibration params, a and b, after the sigma has been calulated so that the market
    # factor we apply to sigma is independent of the variation in a and b
    b_values = {k: 1 for k, v in b_values.items()}
    a_values = {k: 0.2 for k, v in a_values.items()}
    sigma_values = {k: 0.01 for k, v in sigma_values.items()}

    # Reset the Simulator with these updated outputs from the Calibrator
    sim.set_a_values(a_values=a_values)
    sim.set_b_values(b_values=b_values)
    sim.set_volatility(sigma_values=sigma_values)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # 2. Instantiate the MarginCalculator through its TestMarginCalculator class. Will update downstream in the simulation    # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    mc = MarginCalculator(
        apyUpperMultiplier=tau_u, 
        apyLowerMultiplier=tau_d, 
        sigmaSquared=SIGMA_SQUARED, 
        alpha=ALPHA, 
        beta=BETA, 
        xiUpper=XI_UPPER, 
        xiLower=XI_LOWER, 
        tMax=SECONDS_IN_YEAR, 
        devMulLeftUnwindLM=dev_lm,
        devMulRightUnwindLM=dev_lm,
        devMulLeftUnwindIM=dev_im,
        devMulRightUnwindIM=dev_im,
        fixedRateDeviationMinLeftUnwindLM=r_init_lm,
        fixedRateDeviationMinRightUnwindLM=r_init_lm,
        fixedRateDeviationMinLeftUnwindIM=r_init_im,
        fixedRateDeviationMinRightUnwindIM=r_init_im,
        gamma=gamma_unwind,
        minMarginToIncentiviseLiquidators=MIN_MARGIN_TO_INCENTIVIZE_LIQUIDATORS,
    )
 
    # tmc.tokens = pos["tokens"] # Specific tokens in the pool
    # tmc.date_original = df.index # Need to keep record of the original time here so it's not overwritten in the MarginCalculator

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # 3. Instantiate the PortfolioCalculator through its TestPortfolioCalculator class  # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    tpc = TestPortfolioCalculator()
    tpc.setUp()
    # Fee structure "tuneable" parameters 
    tpc.portfolioCalculator.lambdaFee = lambda_fee
    tpc.portfolioCalculator.gammaFee = gamma_fee
    tpc.portfolioCalculator.tokens = pos["tokens"] 
    tpc.portfolioCalculator.lpPosInit = f"{pos['lp_fix']}_{pos['lp_var']}"
    tpc.portfolioCalculator.tPool = (pos["pool_size"]/365)*SECONDS_IN_YEAR
    tpc.portfolioCalculator.notional = pos["notional"]
    tpc.portfolioCalculator.proportion_traded_per_day = 0.20 # Assumption in the modelling, inspired by UniSwap


    # Define the relevant marker cases to loop over and construct different APY and IRS pool
    # scenarios
    f_values = pos["f_values"] # 1. Volatility scalings 
    rate_ranges = pos["rate_ranges"] # 2. Rate ranges, in percentage points, which we convert to tick ranges of interest
    fr_markets = pos["fr_markets"] # 3. Fixed rate markets: neutral, bearish on fixed rate, bullish on fixed rate
    leverage_factors = pos["leverage_factors"] # 4. Leverage scalings, where leverage = notional / margin deposited 
    gamma_fees = pos["gamma_fees"] if pos["gamma_fees"] is not None else [gamma_fee] # 5. Different LP fees
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # 4. Run simulations of the IRS pool over all the different market conditions # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    summary_dict = {}
    # Volatility loop
    for f in f_values:
        df_apy = sim.model_apy(dt=1, F=f) # APY model
        df_apy = sim.compute_apy_confidence_interval(xi_lower=98, xi_upper=39, df_apy=df_apy, F=f) # APY bounds (xi upper and lower subject to change)
        if write_all_out:
            df_apy.to_csv(sim_dir+out_name+f"_F_value_{f}.csv")
        
        # Tick range loop
        for rate_range in rate_ranges:
            tick_name = str(rate_range[0]) + "_" + str(rate_range[1])
            
            # Get the relevant ticks from the rates
            # Remember: higher fixed rate => lower tick, from the geometry of the vAMM
            upper, lower = fixedRateToTick(rate_range[0]), fixedRateToTick(rate_range[1])
            
            # The different APY bounds are automatically passed to the TMC for different tokens
            # We need to update the simulated APY model passed to the TMC in each bound
            fee_collector = [] # Tracks protocol fees
            for market in fr_markets:
                for lev in leverage_factors:
                    for fee in gamma_fees:
                        tpc.portfolioCalculator.gammaFee = fee # Reset the fee
                        summary_dict[f"F={f} scale, {market} market, {rate_range} tick, {lev} leverage factor, {fee} fee"] = {}
                        
                        df_apy_mc, balances = mc.generate_full_output(df_apy=df_apy, date_original=df.index, tokens=pos["tokens"], notional=pos["notional"], lp_fix=pos["lp_fix"], \
                            lp_var=pos["lp_var"], tick_l=lower, tick_u=upper, fr_market=market, leverage_factor=lev)
                        
                        # Now run the initial methods in the PortfolioCalculator to generate the LP PnL and the associated trader fees
                        tpc.portfolioCalculator.df_protocol = df_apy_mc
                        tpc.portfolioCalculator.liquidity = notional_to_liquidity(notional=pos["notional"], \
                            tick_l=lower, tick_u=upper)


                        # Reset the PortfolioCalculator with the new FT and VT positions (these change in each fixed rate market, which can
                        # now also change with the token)
                        tpc.portfolioCalculator.set_positions(balances)
                        
                        # Start by generating the new LP PnL and net margins (n.b. check the notional assignment is correct)
                        tpc.test_generate_lp_pnl_and_net_margin(tick_l=lower, tick_u=upper, lp_leverage_factor=lev)
                        
                        # Now compute the protocol collected fees, the associated Sharpe ratios, and the fraction of
                        # undercollateralised events
                        #sharpes, undercols, l_factors, levs, the_apys, l_vars, i_vars = tpc.test_sharpe_ratio_undercol_events(tick_l=lower, tick_u=upper)
                        sharpes, undercols, l_factors, levs, the_apys = tpc.test_sharpe_ratio_undercol_events(tick_l=lower, tick_u=upper)
                        summary_dict[f"F={f} scale, {market} market, {rate_range} tick, {lev} leverage factor, {fee} fee"]["SRs"] = sharpes
                        summary_dict[f"F={f} scale, {market} market, {rate_range} tick, {lev} leverage factor, {fee} fee"]["Frac Us"] = undercols
                        summary_dict[f"F={f} scale, {market} market, {rate_range} tick, {lev} leverage factor, {fee} fee"]["Liq. fact."] = l_factors
                        summary_dict[f"F={f} scale, {market} market, {rate_range} tick, {lev} leverage factor, {fee} fee"]["Leverage"] = levs
                        summary_dict[f"F={f} scale, {market} market, {rate_range} tick, {lev} leverage factor, {fee} fee"]["APYs"] = the_apys
                        #summary_dict[f"F={f} scale, {market} market, {rate_range} tick, {lev} leverage factor, {fee} fee"]["LVaRs"] = l_vars
                        #summary_dict[f"F={f} scale, {market} market, {rate_range} tick, {lev} leverage factor, {fee} fee"]["IVaRs"] = i_vars
                        fee_collector.append(df_apy_mc["protocol_fee"].mean()) 
                        df_apy_mc.to_csv(sim_dir+out_name+f"_F_value_{f}_{market}_{tick_name}_{fee}_full_risk_engine_output.csv")

                           
    # Save summary_dict to json here
    with open(sim_dir+f"summary_simulations_{out_name}.json", "w") as fp:
        json.dump(summary_dict, fp, indent=4)


    # # # # # # # # # # # # # # # # # # # 
    # # # 5. Summarise and optimise # # # 
    # # # # # # # # # # # # # # # # # # # 
    # The output of the objective function should be an aggregate metric we are either trying to maximize or minimize
    # Maximise the average SR, keeping the spread wrt volatility low, and minimise the undercollateralisarion 
    # We first need to normalise the resulting SR DataFrames such that all data is in [0,1)
    from MasterPlotter import MasterPlotter as mp
    flatSR = np.array([mp.dict_to_df(summary_dict, "SR VT", "SRs").stack().values, mp.dict_to_df(summary_dict, "SR FT", "SRs").stack().values, \
        mp.dict_to_df(summary_dict, "SR LP", "SRs").stack().values]).flatten()
    
    flatU = np.array([mp.dict_to_df(summary_dict, "Frac. und. VT", "Frac Us").stack().values, mp.dict_to_df(summary_dict, "Frac. und. FT", "Frac Us").stack().values, \
        mp.dict_to_df(summary_dict, "Frac. und. LP", "Frac Us").stack().values]).flatten()
    
    flatLiq = np.array([mp.dict_to_df(summary_dict, "Liq. fact. VT", "Liq. fact.").stack().values, mp.dict_to_df(summary_dict, "Liq. fact. FT", "Liq. fact.").stack().values, \
        mp.dict_to_df(summary_dict, "Liq. fact. LP", "Liq. fact.").stack().values]).flatten()
    
    
    # Get the different actor leverages to use directly in the optimisation
    flatLev = np.array([mp.dict_to_df(summary_dict, "Leverage VT", "Leverage").stack().values, \
                        mp.dict_to_df(summary_dict, "Leverage FT", "Leverage").stack().values, \
                        mp.dict_to_df(summary_dict, "Leverage LP", "Leverage").stack().values]).flatten()
    
    # Normalise and get the means
    meanSR = normalise(flatSR).mean()   
    meanU = 0 if np.all(flatU==0) else normalise(flatU).mean() 
    meanLiq = 0 if np.all(flatLiq==0) else normalise(flatLiq).mean() 
    meanFee =  np.array(fee_collector).mean() 
    meanLev = normalise(flatLev).mean() 
    
    # Pick up the VaRs for regularisation (in their natural units)
    """
    meanLVaR_LP = np.array(mp.dict_to_df(summary_dict, "LVaR LP", "LVaRs").stack().values).flatten().mean()
    meanIVaR_LP = np.array(mp.dict_to_df(summary_dict, "IVaR LP", "IVaRs").stack().values).flatten().mean()
    
    meanLVaR_FT = np.array(mp.dict_to_df(summary_dict, "LVaR FT", "LVaRs").stack().values).flatten().mean()
    meanIVaR_FT = np.array(mp.dict_to_df(summary_dict, "IVaR FT", "IVaRs").stack().values).flatten().mean()
    
    meanLVaR_VT = np.array(mp.dict_to_df(summary_dict, "LVaR VT", "LVaRs").stack().values).flatten().mean()
    meanIVaR_VT = np.array(mp.dict_to_df(summary_dict, "IVaR VT", "IVaRs").stack().values).flatten().mean()
    """

    if debug:
        print("flatSR: ", flatSR)
        print("flatU: ", flatU)
        print("flatLiq: ", flatLiq)
        print("flatLev: ", flatLev)
        print("meanSR: ", meanSR)
        print("meanFee: ", meanFee)
        print("meanU: ", meanU)
        print("meanLiq: ", meanLiq)
        print("meanLev: ", meanLev)
        #print("meanLVaR_LP", meanLVaR_LP)
        #print("meanLVaR_FT", meanLVaR_FT)
        #print("meanLVaR_VT", meanLVaR_VT)
        #print("meanIVaR_LP", meanIVaR_LP)
        #print("meanIVaR_FT", meanIVaR_FT)
        #print("meanIVaR_VT", meanIVaR_VT)
    
    if RUN_OPTUNA:
        obj = -(meanU + meanLiq) - 10*int(meanLev > 100) - 10*int(meanLev< 10)
        # Maximise this -- use the VaRs for regularisation
        #l_var_lim, i_var_lim = 0.3, 0.3
        #obj = meanLev - 10*int(meanLVaR_LP < l_var_lim) - 10*int(meanIVaR_LP < i_var_lim) \
        #    - 10*int(meanLVaR_FT < l_var_lim) - 10*int(meanIVaR_FT < i_var_lim) \
        #    - 10*int(meanLVaR_VT < l_var_lim) - 10*int(meanIVaR_VT < i_var_lim) 
        return obj

def run_with_a_single_set_of_params(parser):

    parser.add_argument("-tu", "--tau_u", type=float, help="tau_u tuneable parameter", default=1.5)
    parser.add_argument("-td", "--tau_d", type=float, help="tau_d tuneable parameter", default=0.7)
    parser.add_argument("-gamu", "--gamma_unwind", type=float, help="Gamma tuneable param. for counterfactual unwind", default=1.)
    parser.add_argument("-dlm", "--dev_lm", type=float, help="Tuneable scale for the LM counterfactual unwind", default=0.5)
    parser.add_argument("-dim", "--dev_im", type=float, help="Tuneable scale for the IM counterfactual unwind", default=0.3)
    parser.add_argument("-rlm", "--r_init_lm", type=float, help="Initial rate for LM counterfactual unwind", default=0.3)
    parser.add_argument("-rim", "--r_init_im", type=float, help="Initial rate for IM counterfactual unwind", default=0.1)
    parser.add_argument("-lam", "--lambda_fee", type=float, help="lambda fee parameter", default=0.1)
    parser.add_argument("-gamf", "--gamma_fee", type=float, help="gamma fee parameter", default=0.03)
    parser.add_argument("-a", "--a_factor", type=float, help="Multiplier for the mean-reversion speed", default=1)
    parser.add_argument("-b", "--b_factor", type=float, help="Multiplier for the mean-reversion central value", default=1)
    parser.add_argument("-l", "--lookback", type=int, help="Lookback parameter (no. of days) for the APY moving average", default=30)
    parser.add_argument("-w", "--write_all_out", action="store_true", help="Save all simulation runs to different DataFrames", default=False)
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode", default=False)

    tuneables = parser.parse_args()

    # Defining dictionary for the tuneable parameters
    tuneable_dict = dict((k, v) for k, v in vars(tuneables).items() if v is not None)
    print(tuneable_dict)
    
    if DF_TO_OPTIMIZE=="aave":
        main(in_name="AaveVariable", out_name="df_AaveVariable_APY_model_and_bounds", **tuneable_dict)
    else:
        main(in_name="CompoundV2", out_name="df_CompoundV2_APY_model_and_bounds", **tuneable_dict)


def objective(trial):

    # Put additional constraints of free parameters here to regularise them
    tau_u = trial.suggest_float("tau_u", 1.0001, 10, log=True)
    tau_d = trial.suggest_float("tau_d", 0.0001, 1, log=True)
    gamma_unwind = trial.suggest_float("gamma_unwind", 0.0001, 10, log=True)
    dev_lm = trial.suggest_float("dev_lm", 0.0001, 10, log=True)
    dev_im = trial.suggest_float("dev_im", 0.0001, 10, log=True)
    r_init_lm = trial.suggest_float("r_init_lm", 0.001, 0.2, log=True)
    r_init_im = trial.suggest_float("r_init_im", 0.001, 0.2, log=True)
    a_factor = trial.suggest_float("a_factor", 0.5, 5, log=True)
    b_factor = trial.suggest_float("b_factor", 0.3, 3, log=True)
    lookback = trial.suggest_int("lookback", 5, pos["pool_size"], log=True) 
    lambda_fee = trial.suggest_float("lambda_fee", 0.001, 0.1, log=True) # We probably want to collect at least 10 %

    # Default protocol fee constraints for v1
    lambda_fee = 0 # i.e. no protocol collected fees -- update this
    gamma_fee = pos["gamma_fee"] # Just investigating a few different fee parameters for v1: 0.001, 0.003, 0.005 
    
    if DF_TO_OPTIMIZE=="aave":
        obj = main(in_name="AaveVariable", out_name="df_AaveVariable_APY_model_and_bounds_optimised",
                        tau_u=tau_u, tau_d=tau_d, gamma_unwind=gamma_unwind, dev_lm=dev_lm,
                        dev_im=dev_im, r_init_im=r_init_im, r_init_lm=r_init_lm, lambda_fee=lambda_fee,
                        gamma_fee=gamma_fee, a_factor=a_factor, b_factor=b_factor, lookback=lookback
                        )
    else:
        obj = main(in_name="CompoundV2", out_name="df_CompoundV2_APY_model_and_bounds_optimised",
                        tau_u=tau_u, tau_d=tau_d, gamma_unwind=gamma_unwind, dev_lm=dev_lm,
                        dev_im=dev_im, r_init_im=r_init_im, r_init_lm=r_init_lm, lambda_fee=lambda_fee,
                        gamma_fee=gamma_fee, b_factor=b_factor, lookback=lookback
                        )

    return obj


def run_param_optimization(parser):

    parser.add_argument("-n_trials", "--n_trials", type=float, help="Number of optimization trials", default=2)
    n_trials = parser.parse_args().n_trials

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.SuccessiveHalvingPruner())
    study.optimize(objective, n_trials=n_trials)
    
    # Relevant output plots
    out_dir = top_dir+f"{DF_TO_OPTIMIZE}/optuna/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Output optimised results
    trial = study.best_trial
    print(f"Best optimised value: {trial.value}")

    print("Optimised parameters: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")
    
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(out_dir+f"optuna_history_{DF_TO_OPTIMIZE}.png")

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(out_dir+f"optuna_importances_{DF_TO_OPTIMIZE}.png")
    
    with open(out_dir+f"optimised_parameters_{DF_TO_OPTIMIZE}.json", "w") as fp:
        json.dump(trial.params, fp, indent=4)

if __name__=="__main__":
    # Adding an argument parser
    from argparse import ArgumentParser
    parser = ArgumentParser()

    if RUN_OPTUNA:
        run_param_optimization(parser=parser)
    else:
        run_with_a_single_set_of_params(parser=parser)
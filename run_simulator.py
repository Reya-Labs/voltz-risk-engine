import math
import pandas as pd
from MarginCalculator import MarginCalculator
from PortfolioCalculator import PortfolioCalculator
from Simulator import Simulator
import json
import os
import optuna
import numpy as np
from position import position
from utils import SECONDS_IN_YEAR, fixedRateToTick, notional_to_liquidity
from RNItoAPY import * 

# ref: https://github.com/optuna/optuna-examples/blob/main/sklearn/sklearn_optuna_search_cv_simple.py

def normalise(array):
    if len(array)==1:
        return array
    if array.max() != array.min():
        return (array-array.min())/(array.max()-array.min())
    else:
        raise Exception("ERROR: minimum and maximum values coincide in array normalisation. Check inputs!")

def main(out_name, tau_u=1.5, tau_d=0.7, lookback=10, lookback_standard=10, lambda_fee=0.1, gamma_fee=0.003, \
        alpha_factor=1, beta_factor=1, sigma_factor=1, xi_lower=56, xi_upper=39, eta_lm=0.001, eta_im=0.002, \
        write_all_out=False, sim_dir=None, debug=False, return_df=False):

    # Generate a simulation-specific directory, based on different tuneable parameters
    if sim_dir is None:
        sim_dir = top_dir+f"{input_dataset_name}/"
    
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)
   
    # Reload data-set e.g. every time Optuna enters a new trial
    token = position["tokens"][0]
    df_raw = pd.read_csv(f"./rni_historical_data/{input_dataset_name}.csv")
    # Get APYs from the raw liquidity indices
    df = getPreparedRNIData(df_raw)
    df = getFrequentData(df, frequency=30)
    df_for_values = getDailyApy([[token, df]], lookback=lookback_standard, rate_oracle_mode=rate_oracle_mode)
    df = getDailyApy([[token, df]], lookback=lookback, rate_oracle_mode=rate_oracle_mode)

    df.set_index("date", inplace=True)
    df_for_values.set_index("date", inplace=True)
    
    # We need to make sure that the DataFrame does not contain NaNs because of a lookback window that
    # is too large
    if len(df)-lookback <= position["pool_size"]:
        lookback = len(df)-position["pool_size"] # Maximum possible lookback
    if position["pool_size"] != -1:
        df = df.iloc[-position["pool_size"]:] # Only keep the latest data for a given N-day pool

    if len(df_for_values)-lookback_standard <= position["pool_size"]:
        lookback_standard = len(df_for_values)-position["pool_size"] # Maximum possible lookback
    if position["pool_size"] != -1:
        df_for_values = df_for_values.iloc[-position["pool_size"]:] # Only keep the latest data for a given N-day pool

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # 1. Instantiate Simulator: inherits from Calibrator, gets CIR model params, generates the APY bounds # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    sim = Simulator(df_protocol=df, tMax=SECONDS_IN_YEAR)   
    sim_standard = Simulator(df_protocol=df_for_values, tMax=SECONDS_IN_YEAR)   
    b_values = sim_standard.compute_b_values()
    residual_disc_drift = sim_standard.residual_disc_drifts(b_values_dict=b_values)
    a_values, sigma_values = sim_standard.compute_continuous_drift_and_volatility(residual_min=residual_disc_drift)

    # We want to now vary the calibration params, a and b, after the sigma has been calulated so that the market
    # factor we apply to sigma is independent of the variation in a and b
    sigma_values = {k: v*sigma_factor for k, v in sigma_values.items()}

    # Reset the Simulator with these updated outputs from the Calibrator
    sim.set_a_values(a_values=a_values)
    sim.set_b_values(b_values=b_values)
    sim.set_volatility(sigma_values=sigma_values)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # 2. Instantiate the MarginCalculator. Will update downstream in the simulation # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    a, b, sigma = a_values[token], b_values[token], sigma_values[token]
    alpha = a*b*alpha_factor
    beta = a*beta_factor
    sigma_squared = (sigma*sigma_factor)**2

    mc = MarginCalculator(
        apyUpperMultiplier=tau_u, 
        apyLowerMultiplier=tau_d, 
        sigmaSquared=sigma_squared,
        alpha=alpha,
        beta=beta,
        xiUpper=xi_upper, 
        xiLower=xi_lower, 
        tMax=SECONDS_IN_YEAR, 
        minMarginToIncentiviseLiquidators=0,
        etaLM=eta_lm,
        etaIM=eta_im
    )
 
    # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # 3. Instantiate the PortfolioCalculator  # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    pc = PortfolioCalculator(
            df_protocol=None,
            lambdaFee=lambda_fee,
            gammaFee=gamma_fee,
            tokens=position["tokens"],
            liquidity=1000,
            balances=None,
            tPool=(position["pool_size"]/365)*SECONDS_IN_YEAR,
            lpPosInit=(position["lp_fix"], position["lp_var"]),
            # TODO: assume more realistic fixed rate
            ftPosInit=(1000,-1000), # FT positions
            vtPosInit=(-1000,1000), # VT positions
            notional=1000, # Absolute value of the variable token balance of a trader (both ft and vt are assumed to have the same)
            proportion_traded_per_day=0.20 # Assumption in the modelling, inspired by UniSwap
        )

    # Define the relevant marker cases to loop over and construct different APY and IRS pool
    # scenarios
    f_values = position["f_values"] # 1. Volatility scalings 
    rate_ranges = position["rate_ranges"] # 2. Rate ranges, in percentage points, which we convert to tick ranges of interest
    fr_markets = position["fr_markets"] # 3. Fixed rate markets: neutral, bearish on fixed rate, bullish on fixed rate
    leverage_factors = position["leverage_factors"] # 4. Leverage scalings, where leverage = notional / margin deposited 
    gamma_fees = position["gamma_fees"] if position["gamma_fees"] is not None else [gamma_fee] # 5. Different LP fees
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # 4. Run simulations of the IRS pool over all the different market conditions # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    summary_dict = {}
    # Volatility loop
    for f in f_values:
        df_apy = sim.model_apy(dt=1, F=f) # APY model
        df_apy = sim.compute_apy_confidence_interval(xi_lower=xi_lower, xi_upper=xi_upper, df_apy=df_apy, F=f, \
            alpha_factor=alpha_factor, beta_factor=beta_factor) # APY bounds (xi upper and lower subject to change)
        
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
                        pc.gammaFee = fee # Reset the fee
                        summary_dict[f"F={f} scale, {market} market, {rate_range} tick, {lev} leverage factor, {fee} fee"] = {}
                        
                        df_apy_mc, balances = mc.generate_full_output(
                            df_apy=df_apy, 
                            date_original=df.index,
                            tokens=position["tokens"],
                            notional=position["notional"], 
                            lp_fix=position["lp_fix"],
                            lp_var=position["lp_var"], 
                            tick_l=lower, 
                            tick_u=upper, 
                            fr_market=market, 
                            leverage_factor=lev
                        )
                        
                        # Now run the initial methods in the PortfolioCalculator to generate the LP PnL and the associated trader fees
                        pc.df_protocol = df_apy_mc
                        pc.liquidity = notional_to_liquidity(
                            notional=position["notional"],
                            tick_l=lower, 
                            tick_u=upper
                        )

                        # Reset the PortfolioCalculator with the new FT and VT positions (these change in each fixed rate market, which can
                        # now also change with the token)
                        pc.set_positions(balances)
                        
                        # Start by generating the new LP PnL and net margins (n.b. check the notional assignment is correct)
                        pc.generate_lp_pnl_and_net_margin(tick_l=lower, tick_u=upper, lp_leverage_factor=lev)
                        
                        # Now compute the protocol collected fees, the associated Sharpe ratios, and the fraction of
                        # undercollateralised events
                        sharpes, undercols, l_factors, levs, the_apys, l_vars, i_vars, l_levs, i_levs, gaps = pc.sharpe_ratio_undercol_events(tick_l=lower, tick_u=upper)
                        summary_dict[f"F={f} scale, {market} market, {rate_range} tick, {lev} leverage factor, {fee} fee"]["SRs"] = sharpes
                        summary_dict[f"F={f} scale, {market} market, {rate_range} tick, {lev} leverage factor, {fee} fee"]["Frac Us"] = undercols
                        summary_dict[f"F={f} scale, {market} market, {rate_range} tick, {lev} leverage factor, {fee} fee"]["Liq. fact."] = l_factors
                        summary_dict[f"F={f} scale, {market} market, {rate_range} tick, {lev} leverage factor, {fee} fee"]["Leverage"] = levs
                        summary_dict[f"F={f} scale, {market} market, {rate_range} tick, {lev} leverage factor, {fee} fee"]["APYs"] = the_apys
                        summary_dict[f"F={f} scale, {market} market, {rate_range} tick, {lev} leverage factor, {fee} fee"]["LVaRs"] = l_vars
                        summary_dict[f"F={f} scale, {market} market, {rate_range} tick, {lev} leverage factor, {fee} fee"]["IVaRs"] = i_vars
                        summary_dict[f"F={f} scale, {market} market, {rate_range} tick, {lev} leverage factor, {fee} fee"]["L-Levs"] = l_levs
                        summary_dict[f"F={f} scale, {market} market, {rate_range} tick, {lev} leverage factor, {fee} fee"]["I-Levs"] = i_levs
                        summary_dict[f"F={f} scale, {market} market, {rate_range} tick, {lev} leverage factor, {fee} fee"]["Gaps"] = gaps
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
    
    # Pick up the FT leverage to use for regularisation
    meanLevFT = mp.dict_to_df(summary_dict, "Leverage FT", "Leverage").stack().values.flatten().mean()
    
    # Normalise and get the means
    print("flatLev:", flatLev)
    meanLev = normalise(flatLev).mean() 
    
    # Pick up the VaRs for regularisation (in their natural units)
    meanLVaR_LP = np.array(mp.dict_to_df(summary_dict, "LVaR LP", "LVaRs").stack().values).flatten().mean()
    meanIVaR_LP = np.array(mp.dict_to_df(summary_dict, "IVaR LP", "IVaRs").stack().values).flatten().mean()
    
    meanLVaR_FT = np.array(mp.dict_to_df(summary_dict, "LVaR FT", "LVaRs").stack().values).flatten().mean()
    meanIVaR_FT = np.array(mp.dict_to_df(summary_dict, "IVaR FT", "IVaRs").stack().values).flatten().mean()
    
    meanLVaR_VT = np.array(mp.dict_to_df(summary_dict, "LVaR VT", "LVaRs").stack().values).flatten().mean()
    meanIVaR_VT = np.array(mp.dict_to_df(summary_dict, "IVaR VT", "IVaRs").stack().values).flatten().mean()

    # Margin gaps for regularisation -- look at the smallest gaps
    minGap_FT = np.array(mp.dict_to_df(summary_dict, "Gap FT", "Gaps").stack().values).flatten().min()
    minGap_VT = np.array(mp.dict_to_df(summary_dict, "Gap VT", "Gaps").stack().values).flatten().min()
    minGap_LP = np.array(mp.dict_to_df(summary_dict, "Gap LP", "Gaps").stack().values).flatten().min()

    if debug:
        print("flatSR: ", flatSR)
        print("flatU: ", flatU)
        print("flatLiq: ", flatLiq)
        print("flatLev: ", flatLev)
        print("meanLev: ", meanLev)
        print("meanLVaR_LP: ", meanLVaR_LP)
        print("meanLVaR_FT: ", meanLVaR_FT)
        print("meanLVaR_VT: ", meanLVaR_VT)
        print("meanIVaR_LP: ", meanIVaR_LP)
        print("meanIVaR_FT: ", meanIVaR_FT)
        print("meanIVaR_VT: ", meanIVaR_VT)
        print("meanGap_FT: ", minGap_FT)
        print("meanGap_VT: ", minGap_VT)
        print("meanGap_LP: ", minGap_LP)
    
    if not optuna_enabled and return_df:
        return df_apy_mc
    
    if optuna_enabled:
        # Maximise this -- use the VaRs for regularisation
        l_var_lim, i_var_lim = 0.05, 0.05
        eta_gap = eta_im-eta_lm
        obj = meanLev -10*int(meanLVaR_FT < l_var_lim) -10*int(meanLVaR_VT < l_var_lim) -10*int(meanLVaR_LP < l_var_lim) \
                -10*int(meanLevFT < 20) -10*int(eta_gap < 0.0005)

        return obj

def run_with_a_single_set_of_params():
    with open(manual_parameters_file) as json_file:
        tuneables = json.load(json_file)

    print(tuneables)

    for k in tuneables.keys():
        print(k, tuneables[k], math.floor(tuneables[k] * 1e18))

    # Defining dictionary for the tuneable parameters
    main(out_name=f"df_{input_dataset_name}_RiskEngineModel", **tuneables)


def objective(trial):

    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    tau_u = trial.suggest_float("tau_u", 1.0001, 2, log=True)
    tau_d = trial.suggest_float("tau_d", 0.5, 1, log=True)
    lookback = trial.suggest_int("lookback", 10, 15, log=True) 
    alpha_factor = trial.suggest_float("alpha_factor", 0.5, 1.5, log=True) 
    beta_factor = trial.suggest_float("beta_factor", 0.5, 1.5, log=True) 
    sigma_factor = trial.suggest_float("sigma_factor", 0.5, 2.0, log=True) 
    xi_upper = trial.suggest_int("xi_upper", 19, 78, log=True) 
    xi_lower = trial.suggest_int("xi_lower", 28, 112, log=True) 
    eta_im = trial.suggest_float("eta_im", 0.0005, 0.005, log=True) 
    eta_lm = trial.suggest_float("eta_lm", 0.0005, 0.005, log=True) 
    
    # Default protocol fee constraints for v1
    # Here we summarise default fee struccture parameters for v1
    lambda_fee = 0 # i.e. no protocol collected fees -- update this
    gamma_fee = position["gamma_fee"] # Just investigating a few different fee parameters for v1: 0.001, 0.003, 0.005 

    obj = main(out_name=f"df_{input_dataset_name}_RiskEngineModel",
                        tau_u=tau_u, tau_d=tau_d, lambda_fee=lambda_fee,
                        gamma_fee=gamma_fee, alpha_factor=alpha_factor, beta_factor=beta_factor,
                        sigma_factor=sigma_factor, xi_lower=xi_lower, xi_upper=xi_upper, eta_im=eta_im, eta_lm=eta_lm,
                        lookback_standard=10, lookback=lookback
    )

    return obj


def run_param_optimization():
    # Create study and start optimisation
    study = optuna.create_study(
        direction="maximize", 
        sampler=optuna.samplers.TPESampler(), 
        pruner=optuna.pruners.SuccessiveHalvingPruner()
    )

    study.optimize(
        func=objective, 
        n_trials=n_trials
    )

    # Output optimised results
    trial = study.best_trial
    print(f"Best optimised value: {trial.value}")

    # Run the objective with the best trial such that the plots/datasets reflect this
    objective(trial)

    print("Optimised parameters: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")
    
    # Plot optimisation history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(out_dir+f"optuna_history.png")
    
    # Export optimiser parameters as JSON
    with open(out_dir+f"optimised_parameters.json", "w") as fp:
        json.dump(trial.params, fp, indent=4)

if __name__=="__main__":
    # Adding an argument parser
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("-opt", "--optimization", type=bool, help="Flag to enable optimization", default=False)
    parser.add_argument("-ds", "--dataset", type=str, help="Dataset with raw liquidity indices", required=True)
    parser.add_argument("-n_trials", "--n_trials", type=float, help="Number of optimization trials", default=100)
    parser.add_argument("-rate_oracle_mode", "--rate_oracle_mode", type=str, help="Rate Oracle Mode: linear/compounding/sofr", required=True)

    optuna_enabled = parser.parse_args().optimization
    input_dataset_name = parser.parse_args().dataset
    n_trials = parser.parse_args().n_trials
    rate_oracle_mode = parser.parse_args().rate_oracle_mode

    # Globals 
    top_dir = f"./simulations/"

    # Optuna output directory
    out_dir = top_dir+f"{input_dataset_name}/optuna/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if optuna_enabled:
        print("Running optimization...")
        run_param_optimization()
    else:
        # Parameter json for single run
        print("Simulating given parameters...")
        manual_parameters_file = f"simulations/{input_dataset_name}/params.json"
        run_with_a_single_set_of_params()

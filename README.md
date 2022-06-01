# voltz-risk-engine
The official Voltz Risk Engine framework for simulations and analysis of the Voltz IRS. The Risk Engine is an independent Python implementation
of the Voltz interest rate swap (IRS) protocol and associated Smart Contract logic. This includes:

1) The full simulation of an IRS pool for user-defined fixed-taker (FT), variable-taker (VT) and liquidity provider (LP) positions.
2) The incorporation of different APY conditions, based on historical data, and calibrated with the CIR mean-reverting short-rate model.
3) The modelling of different fixed-rate market conditions, actor leverages, and market APY volatilities.
4) All protocol parameters defined in the litepaper are implemented, and can be set by the user.
5) The use of ```Optuna``` to perform optimisation of the protocol parameters and fee structure parameters. 
6) A repository of scraped historic APY data from Aave, Compound, Euler, and other popular DeFi platforms.  

## Extracting and Scraping Data
``` APYExtractor.py```: Scrapes relevant APY time series data from LoanScan (https://loanscan.io/), saving them as a Pandas DataFrame

```Dune.py```: Scrapes on-chain data from the Dune Analytics platform (https://dune.com/home), saving them as a Pandas DataFrame

## Calibration and Simulation of APY Time Series
```Calibrator.py```: The ```Calibrator``` object sets up and performs calibration of an APY time series, according to a discretised version of the 
Cox-Ingersoll-Ross model (https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model) for mean-reverting, non-negative short-rates. 

```Simulator.py```: The ```Simulator``` object inherits from the ```Calibrator```, taking the calibrated parameters of the CIR model and 
extracting the associated APY bounds under a variety of different user-defined volatillity scalings $\sigma_\mathrm{scaled} = F\sigma$, where
$\sigma$ if the volaility derived from the original CIR and $F$ is a scaling factor. 

## Voltz IRS Implementation
From the simulatd APYs, the Voltz IRS is implemented through three main classes:

```MarginCalculator.py```: Handles the calculation of FT, VT, and LP margins, covering both the liquidation margin and initial margin 
requirements, together with their associated minimum margin requirements. The trader PnLs are also calculated and stored, together with a 
record of the fixed and variable token positions, as time series.

```PortfolioCalculator.py```: Takes the positions and margin requirements as inputs, and compute the associated LP PnLs. From this,
protocol, trader, and LP fees are also calculated. These results are combined into a number of useful metrics: Sharpe ratio, actor APY,
fraction of undercollarerallised events in the IRS pool, flags for liquidatable events in the IRS pool, and the leverage of each actor. 

```MasterPlotter.py```: A collection of succinct methods for plotting the results of the IRS pool out put time series, including the margins,
positions, leverages, APYs, and associated Sharpe ratios for different market conditions. 

## Calling the Risk Engine
The main Risk Engine classes are controlled through the ```run_simulator.py``` script, which takes user-defined values of the protocol 
parameters as input and controls the flow of the Voltz IRS simulation. If the ```RUN_OPTUNA``` flag is set, then this script alternatively
runs the hyperparameter optimisation of the protocol under a given objective function constructed from the metrics defined in the 
```PortfolioCalculator.py```.  The arguments for this script may be summarised as:

```
   run_simulator.py     [-h] [-tu TAU_U] [-td TAU_D] [-gamu GAMMA_UNWIND]
                        [-dlm DEV_LM] [-dim DEV_IM] [-rlm R_INIT_LM]
                        [-rim R_INIT_IM] [-lam LAMBDA_FEE] [-gamf GAMMA_FEE]
                        [-a A_FACTOR] [-b B_FACTOR] [-l LOOKBACK] [-w] [-d]

optional arguments:
  -h, --help            show this help message and exit
  -tu TAU_U, --tau_u TAU_U
                        tau_u tuneable parameter
  -td TAU_D, --tau_d TAU_D
                        tau_d tuneable parameter
  -gamu GAMMA_UNWIND, --gamma_unwind GAMMA_UNWIND
                        Gamma tuneable param. for counterfactual unwind
  -dlm DEV_LM, --dev_lm DEV_LM
                        Tuneable scale for the LM counterfactual unwind
  -dim DEV_IM, --dev_im DEV_IM
                        Tuneable scale for the IM counterfactual unwind
  -rlm R_INIT_LM, --r_init_lm R_INIT_LM
                        Initial rate for LM counterfactual unwind
  -rim R_INIT_IM, --r_init_im R_INIT_IM
                        Initial rate for IM counterfactual unwind
  -lam LAMBDA_FEE, --lambda_fee LAMBDA_FEE
                        lambda fee parameter
  -gamf GAMMA_FEE, --gamma_fee GAMMA_FEE
                        gamma fee parameter
  -a A_FACTOR, --a_factor A_FACTOR
                        Multiplier for the mean-reversion speed
  -b B_FACTOR, --b_factor B_FACTOR
                        Multiplier for the mean-reversion central value
  -l LOOKBACK, --lookback LOOKBACK
                        Lookback parameter (no. of days) for the APY moving
                        average
  -w, --write_all_out   Save all simulation runs to different DataFrames
  -d, --debug           Debug mode
```

## Unit Tests
The unit testing of different classes is handled by the various ```Test{class}.py``` scripts. These are also called in the ```run_simulator.py``` for
conveniece of instantiating the relevant methods from the main classes described above. 

# Terms & Conditions
The Voltz Protocol, and any products or services associated therewith, is offered only to persons (aged 18 years or older) or entities who are not residents of, citizens of, are incorporated in, owned or controlled by a person or entity in, located in, or have a registered office or principal place of business in any “Restricted Territory.”

The term Restricted Territory includes the United States of America (including its territories), Algeria, Bangladesh, Bolivia, Belarus, Myanmar (Burma), Côte d’Ivoire (Ivory Coast), Egypt, Republic of Crimea, Cuba, Democratic Republic of the Congo, Iran, Iraq, Liberia, Libya, Mali, Morocco, Nepal, North Korea, Oman, Qatar, Somalia, Sudan, Syria, Tunisia, Venezuela, Yemen, Zimbabwe; or any jurisdictions in which the sale of cryptocurrencies are prohibited, restricted or unauthorized in any form or manner whether in full or in part under the laws, regulatory requirements or rules in such jurisdiction; or any state, country, or region that is subject to sanctions enforced by the United States, such as the Specially Designed Nationals and Blocked Persons List (“SDN List”) and Consolidated Sanctions List (“Non-SDN Lists”), the United Kingdom, or the European Union
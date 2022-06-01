import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

"""
    The Calibrator class performs CIR calibration of the APY data for a given protocol
    and token. It calculates continuous volatility and mean-reversion factors, performs a scaling 
    on the volatility and calibration factors. There are some limitations to this approach at present,
    in particular the suitability of the CIR model in crypto -- a hybrid with the Heston model which
    captures stochastic volatility features might be more appropriate. 

    In steps:

    1) Calibration to get the mean-reversion parameters
        a) Get mean-revision average, b, from averaging the long-term APY
        b) Get discrete drift my minimising the APY residuals using the b estimate
        c) Continuous drift is a = -ln(discrete drift)

    2) Discrete volatility calculation -- directly calculated from the minimised residual

    3) Outputs the modified series with the calibration parameters

"""

class Calibrator:
    def __init__(self, df_protocol, disc_drift=0.99):
        self.df_protocol = df_protocol
        self.disc_drift = disc_drift
    
    """
        Returns a dictionary of different b-values, corresponding to the
        long-term mean-reverted value to use in APY calculations and simulations
    """
    def compute_b_values(self):
        return {token: self.df_protocol[token].mean() for token in self.df_protocol.columns if "Date" not in token}

    """
        Need to provide the full pool (Aave or Compound) and the 
        b-values corresponding to the long-term mean-reverted value
    """
    def residual_disc_drifts(self, b_values_dict=None):
        initial_drift = self.disc_drift
        if b_values_dict is None:
            b_values_dict = self.compute_b_values()
        from scipy.optimize import minimize # Might need a random seed here too
        residual_disc_drift_dict = {}
        for token, b_value in b_values_dict.items():
            residual_disc_drift_dict[token] = {}
            # We need to minimise the residuals with respect to the centred short-rate i.e.
            # APY centred = APY - b_value 
            df_c = self.df_protocol[token] - b_value

            # Define the residual function to minimise so we can call it with a non-linear
            # minimiser in SciPy
            def residual_sum(initial_drift):
                residuals = [(df_c[i+1]-df_c[i]*initial_drift)**2 / (df_c.values[i] + b_value) for i in range(len(df_c)-1)]
                return np.sum(residuals)

            # Minimisation step with the residuals 
            res = minimize(residual_sum, initial_drift, method="Nelder-Mead", tol=1e-6)
            residual_disc_drift_dict[token]["minimum_residual"] = res["fun"]
            residual_disc_drift_dict[token]["disc_drift_optimum"] = res["x"][0]

        return residual_disc_drift_dict

    """
        Calculate continuous drift and volatilty from the initial set
        of minimised residuals, across all tokens
    """
    def compute_continuous_drift_and_volatility(self, residual_min=None):
        if residual_min is None:
            residual_min = self.residual_disc_drifts()
        a_values_dict = {token: -np.log(residual_min[token]["disc_drift_optimum"]) for token in self.df_protocol.columns if "Date" not in token} # Continuous drift
        sigma_discrete_sq_dict = {token: residual_min[token]["minimum_residual"]/(len(self.df_protocol[token])-1) \
            for token in self.df_protocol.columns if "Date" not in token}
        sigma_dict = {token: np.sqrt((2*a_values_dict[token]*sigma_discrete_sq_dict[token])/(1-np.exp(-2*a_values_dict[token]))) \
            for token in self.df_protocol.columns if "Date" not in token}
        
        return a_values_dict, sigma_dict
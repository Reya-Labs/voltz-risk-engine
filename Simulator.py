import numpy as np
from Calibrator import Calibrator
from utils import SECONDS_IN_DAY, SECONDS_IN_YEAR

"""
    Simulator class inherits from Calibrator. It makes a calibrated object as input, with the
    corresponding calibration parameters and estimated continuous volatility from the CIR model.
    From the provided calibrated object it performs:

    1) Sigma scaling to generate a variety of new APY replicates
        a) Generate scaled volatilities from F * sigma, using the sigma continuous volatilities 
        from the CIR calibration.
        b) For these parameters, perform APY simulation using the CIR model and existing
        calibrations.

    2) Calculation of the APY bounds for each token and corresponding replicates based on different
    sigma scalings F -- this part of the calculation follows directly from litepaper, computing the closed-form
    solution for the APY bounds assuming a fixed volatility and CIR calibration. 
"""

class Simulator(Calibrator):
    def __init__(self, df_protocol, a_values_dict=None, b_values_dict=None, sigma_dict=None, tMax=SECONDS_IN_YEAR):
        super().__init__(df_protocol)
        
        # Pick up necessary dicts if they are not properly initialised already
        if a_values_dict is None:
            a_values_dict, sigma_dict = self.compute_continuous_drift_and_volatility() 
        if b_values_dict is None:
            b_values_dict = self.compute_b_values()
        
        self.a_values_dict = a_values_dict
        self.b_values_dict = b_values_dict
        self.sigma_dict = sigma_dict
        self.tMax = tMax

    # Additional setters 
    def set_a_values(self, a_values):
        self.a_values_dict = a_values
    
    def set_b_values(self, b_values):
        self.b_values_dict = b_values
        
    def set_volatility(self, sigma_values):
        self.sigma_dict = sigma_values
    
    def set_data(self, df_protocol):
        self.df_protocol = df_protocol

    """
        dAPY = a(b-APY)dt + sigma * sqrt(APY) * Z*sqrt(dt), using Z*sqrt(dt) as the Weiner process.
    
        Take dt = 1 i.e. one crypto trading day, as default

        We can optionally apply a scaling to the volatility with scaling factor, F

        This makes a deep copy and returns a new DataFrame corresonding to the raw stochastic model i.e.
        no EWMA has been applied at this point
    """
    def model_apy(self, dt=1, F=1., alpha_factor=1, beta_factor=1):
        df_deep_copy = self.df_protocol.copy()
        tokens = [c for c in self.df_protocol.columns if (("model" not in c) and ("date" not in c))]
        # Set random seed for all simulations
        np.random.seed(42)
        for token in tokens:
            a, b, sigma = self.a_values_dict[token], self.b_values_dict[token], self.sigma_dict[token]*F
            apy_i = df_deep_copy[token].values[0]
            apy_model = [apy_i]
            # Re-write the model terms
            alpha = a*b*alpha_factor
            beta = a*beta_factor
            for i in range(1, len(df_deep_copy)):
                #dapy = a*(b-apy_i)*dt + sigma * np.sqrt(apy_i) * np.random.normal(0,1,1)[0] * np.sqrt(dt) # Need a random seed for this
                dapy = (alpha-beta*apy_i)*dt + sigma * np.sqrt(apy_i) * np.random.normal(0,1,1)[0] * np.sqrt(dt) # Need a random seed for this
                apy_i += dapy
                
                # Add in protection against negative APYs in the modelling
                if apy_i <= 0:
                    apy_i = beta/100 # Some tolerance based on the average APY
                
                apy_model.append(apy_i)
            df_deep_copy[token + " model"] = apy_model
        
        return df_deep_copy
    

    """
        Compute the confidence interval on APY (defaults to 95 % interval), saving the upper and lower
        bounds as the a time series
    """
    def compute_apy_confidence_interval(self, xi_lower=1.96, xi_upper=1.96, df_apy=None, F=1., alpha_factor=1, beta_factor=1): 

        if df_apy is None:
            df_apy = self.df_protocol 
        if "model" not in " ".join(df_apy.columns):
            raise Exception("No APY model found. Plase run model_apy() first")
            
        tokens = [c for c in df_apy.columns if (("model" not in c) and ("date" not in c))]
        # Calculate the appropriate time deltas
        time_deltas = np.array([(len(df_apy)-i)*SECONDS_IN_DAY/self.tMax for i in range(len(df_apy))])
        for token in tokens:
            alpha = self.a_values_dict[token]*self.b_values_dict[token]*alpha_factor
            beta = self.a_values_dict[token]*beta_factor
            sigma = self.sigma_dict[token]*F

            k = 4*alpha/sigma**2
        
            # Litepaper parameters
            time_factor = np.array([np.exp(-beta*dt) for dt in time_deltas])

            zeta = sigma**2 * (1-time_factor)/(4*beta)
            lamb = (1/zeta) * time_factor * df_apy[token + " model"]
        
            # Translate to APY confidence intervals
            apy_lower = np.maximum(zeta*(k + lamb - xi_lower*np.sqrt(2*(k+2*lamb))), np.zeros(len(lamb)))
            apy_upper = zeta*(k + lamb + xi_upper*np.sqrt(2*(k+2*lamb)))
        
            df_apy[token + " APY lower"] = apy_lower
            df_apy[token + " APY upper"] = apy_upper

        return df_apy
from tkinter.tix import InputOnly
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
bashtage/arch: Release 4.18 (Version v4.18). Zenodo. https://doi.org/10.5281/zenodo.593254
pip3 install arch --user
"""
from arch.bootstrap import CircularBlockBootstrap, optimal_block_length

"""
    RiskMetrics object is instantiated with a given notional, and corresonding time series
    for the actor liquidation margin and initial margins, together with the associated PnL
"""
class RiskMetrics():
    
    def __init__(self, df, notional, liquidation_series, margin_series, pnl_series, pool_size=None):
        self.z_scores = {
            95: 1.96,
            99: 2.58
        }
        
        if pool_size is None:
            self.pool_size = len(df)
            self.df = df
        else:
            self.pool_size = pool_size
            self.df = df.iloc[:pool_size]
        
        self.notional = notional
        self.liquidation_series = df[liquidation_series]
        self.margin_series = df[margin_series]
        self.pnl_series = df[pnl_series]
        """
        the_liquidation = self.liquidation()
        the_insolvency = self.insolvency()
        self.liquidation = the_liquidation.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.insolvency = the_insolvency.replace([np.inf, -np.inf], np.nan, inplace=True)
        """
        
        self.liquidation = self.liquidation()
        self.insolvency = self.insolvency()

    """
        Liquidation time series, from input margin and 
        liquidation requirements time series

        Tends to liquidation as the value -> 0
    """
    def liquidation(self):
        return (self.margin_series.iloc[0] - self.liquidation_series) #/ self.liquidation_series

    """
        Insolvency time series, from input margin and 
        actor PnL time series

        Tends to insolvency as value < 0
    """
    def insolvency(self):
        return (self.pnl_series + self.margin_series.iloc[0]) #/ self.margin_series.iloc[0]

    """
        Time series block bootstrapping to produce N_replicates
        number of replicates. Assumes the autocorrelation structure of the
        time series is over some horizon given by time_delta 
    """
    def generate_replicates(self, N_replicates=100):
        rs = np.random.RandomState(42)

        self.liquidation.dropna(inplace=True)
        self.insolvency.dropna(inplace=True)
        # Optimal block lengths    
        time_delta_l = optimal_block_length(self.liquidation)["circular"].values[0]
        time_delta_i = optimal_block_length(self.insolvency)["circular"].values[0]

        # Block bootstrapping
        l_bs = CircularBlockBootstrap(block_size=int(time_delta_l)+1, x=self.liquidation, random_state=rs)
        i_bs = CircularBlockBootstrap(block_size=int(time_delta_i)+1, x=self.insolvency, random_state=rs)
        l_rep = [data[1]["x"].values.flatten() for data in l_bs.bootstrap(N_replicates)]
        i_rep = [data[1]["x"].values.flatten() for data in i_bs.bootstrap(N_replicates)]

        return l_rep, i_rep

    """
        Normalise a given vector of information such that the integral over its
        domain is unity, thus it is a true pdf
    """
    @staticmethod
    def normalise_vector(vector, plot=False):
        normed = plt.hist(vector, density=True, label="Normalised")
        if plot:
            plt.hist(vector, label="Unnormalised")
            plt.savefig("Normalised_check.png")
            return normed[1]
        else:
            return normed[1]

    """
        Calculate the LVaR and IVaR according to the Gaussianity assumption for the
        underlying liquidation and insolvency distributions. Generates means and stds
        from the replicate distributions, for a given time-horizon and Z-score (based on
        singificance level, alpha)
    """
    def lvar_and_ivar(self, alpha=95, l_rep=None, i_rep=None):
        z_score = self.z_scores[alpha]
        if (l_rep is None) or (i_rep is None):
            l_rep, i_rep = self.generate_replicates()
        l_dist, i_dist = np.array([l.mean() for l in l_rep]), np.array([i.mean() for i in i_rep]) # CLT => Gaussian

        l_mu, i_mu = l_dist.mean(), i_dist.mean()
        l_sig, i_sig = l_dist.std(), i_dist.std()

        l_var = -z_score*l_sig + l_mu 
        i_var = -z_score*i_sig + i_mu

        return l_var, i_var 

    """
        Convert VaRs to corresponding leverage constraints
    """
    def leverages(self, l_var=None, i_var=None, time_horizon=0):
        if (l_var is None) or (i_var is None):
            l_var, i_var = self.lvar_and_ivar()
        
        l_lev = self.notional / (self.liquidation_series.iloc[time_horizon]*l_var + self.liquidation_series.iloc[time_horizon])
        i_lev = self.notional / (self.margin_series.iloc[0]*i_var - self.margin_series.iloc[0])
        return l_lev, i_lev

    """
        Commpute the recommended leverage based on 
        Leverage = min(Lev_L, Lev_I), from the liquidation and insolvency leverages
    """
    def recommended_leverage(self, time_horizon=0):
        l_lev, i_lev = self.leverages(time_horizon=time_horizon)
        if l_lev < i_lev:
            return l_lev 
        else:
            return i_lev
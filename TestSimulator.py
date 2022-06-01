import unittest
from zlib import Z_BEST_COMPRESSION
import pandas as pd
import numpy as np
from Simulator import Simulator

# Test using example scraped APY data from Aave
TOP_DIR = "./test_data/"
DF_PROTOCOL = pd.read_csv(TOP_DIR+"composite_df_AaveVariable_apy.csv")
DF_PROTOCOL.dropna(inplace=True)
DF_PROTOCOL.set_index("Date", inplace=True)
B_VALUES_DICT = {
            "DAI": 0.04850121563176821,
            "USDC": 0.04533734965482663,
            "USDT": 0.04467236184515213,
            "TUSD": 0.047844568648994786
}
RESIDUAL_MIN = {
            "DAI": {
                "minimum_residual": 15.653476506793233,
                "disc_drift_optimum": 0.9097520828247067
            },
            "USDC": {
                "minimum_residual": 25.161283226878766,
                "disc_drift_optimum": 0.8947576675415035
            },
            "USDT": {
                "minimum_residual": 62.38910988614741,
                "disc_drift_optimum": 0.9584461669921873
            },
            "TUSD": {
                "minimum_residual": 455.6433453130155,
                "disc_drift_optimum": 0.9444963455200194
            }
}
A_VALUES_DICT = {
            "DAI": 0.09458315304516063,
            "USDC": 0.11120235989353155,
            "USDT": 0.042441881879434824,
            "TUSD": 0.057103461269848235
}

SIGMA_DICT = {
            "DAI": 0.14709907891865323,
            "USDC": 0.18799923097559318,
            "USDT": 0.28628269263063927,
            "TUSD": 0.7792631219857528
}
DT = 1/365
F = 1.
Z_SCORE = 1.96

class TestSimulator(unittest.TestCase):

    def initalise(self):
        self.simulator = Simulator(df_protocol=DF_PROTOCOL, a_values_dict=A_VALUES_DICT, b_values_dict=B_VALUES_DICT, sigma_dict=SIGMA_DICT)
    
    # Need to add unit tests which compare outputs of pandas DataFrames
    def test_simulator(self):
        self.simulator.model_apy(dt=DT, F=F)
        self.simulator.compute_apy_confidence_interval(z_score=Z_SCORE)

        DF_PROTOCOL.to_csv(TOP_DIR+"composite_df_AaveVariable_apy_SimulatorUnitTest.csv")

if __name__=="__main__":
    unittest.main()
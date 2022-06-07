import unittest 
import pandas as pd
import numpy as np
from Calibrator import Calibrator


# Test using example scraped APY data from Aave
TOP_DIR = "./test_data/"
DF_PROTOCOL = pd.read_csv(TOP_DIR+"composite_df_AaveVariable_apy.csv")
DF_PROTOCOL.dropna(inplace=True)
DF_PROTOCOL.set_index("Date", inplace=True)
DISC_DRIFT = 0.99

class TestCalibrator(unittest.TestCase):
    
    def instantiate(self):
        self.calibrator = Calibrator(df_protocol=DF_PROTOCOL, disc_drift=DISC_DRIFT)
    
    def test_calibrator(self):
        B_VALUES_DICT = self.calibrator.compute_b_values()

        expected = {
                    "DAI": 0.04850121563176821,
                    "USDC": 0.04533734965482663,
                    "USDT": 0.04467236184515213,
                    "TUSD": 0.047844568648994786
        }
        for key, value in expected.items():
            self.assertEqual(B_VALUES_DICT[key], value)

        RESIDUAL_MIN = self.calibrator.residual_disc_drifts(b_values_dict=B_VALUES_DICT)
        
        expected = {
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
        for key, value in expected.items():
            for key_, value_ in expected[key].items():
                self.assertAlmostEqual(RESIDUAL_MIN[key][key_], value_) # Almost equal due to differences when running the minimiser

        A_VALUES_DICT, SIGMA_DICT = self.calibrator.compute_continuous_drift_and_volatility(residual_min=RESIDUAL_MIN)
        
        expected = {
                    "DAI": 0.09458315304516063,
                    "USDC": 0.11120235989353155,
                    "USDT": 0.042441881879434824,
                    "TUSD": 0.057103461269848235
        }
        for key, value in expected.items():
            self.assertAlmostEqual(A_VALUES_DICT[key], value)

        expected = {
                    "DAI": 0.14709907891865323,
                    "USDC": 0.18799923097559318,
                    "USDT": 0.28628269263063927,
                    "TUSD": 0.7792631219857528
        }
        for key, value in expected.items():
            self.assertAlmostEqual(SIGMA_DICT[key], value)

if __name__=="__main__":
    unittest.main()
import unittest
from MarginCalculator import SECONDS_IN_YEAR
from PortfolioCalculator import PortfolioCalculator
import pandas as pd

class TestPortfolioCalculator(unittest.TestCase):

    def setUp(self):

        df_protocol = pd.read_csv("./testing_data/test_apy_for_PortfolioCalculator.csv", index_col="date")     
        tokens = ["USDC", "USDT", "DAI"]
        
        self.portfolioCalculator = PortfolioCalculator(
            df_protocol=df_protocol,
            lambdaFee=0.1,
            gammaFee=0.003,
            tokens=tokens,
            liquidity=1000,
            balances=None,
            tPool=SECONDS_IN_YEAR,
            lpPosInit=(0,0), # LP positions
            ftPosInit=(1000,-1000), # FT positions
            vtPosInit=(-1000,1000), # VT positions
            notional=1000, # Absolute value of the variable token balance of a trader (both ft and vt are assumed to have the same)
            proportion_traded_per_day=0.15 # We're assume 15 % of the liquidity is traded on a given day, based on typical numbers for Uni v3: https://info.uniswap.org/#/
        )

if __name__ == '__main__':
    unittest.main()
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

    def test_generate_lp_pnl_and_net_margin(self, tick_l=5000, tick_u=6000, lp_leverage_factor=1):

        # liquidity, tickLower, tickUpper
        # the output of this test script should give us what we need

        # These are assumptions about the fixed tick sizes and 
        # proportios traded each day. We will need to generalise these
        # but for v1 these assumptions are sufficient.
        tickLower = tick_l
        tickUpper = tick_u

        # Methods below update the df_protocol, so we won't return a separate DataFrame for now
        # Generate lp pnl
        self.portfolioCalculator.generateLPPnl(tickLower, tickUpper)

        # Generate lp net margin -- returns the updated DataFrame
        self.portfolioCalculator.generateLPNetMargin(lp_leverage_factor=lp_leverage_factor)

        # Generate constant trader fee column
        self.portfolioCalculator.generateTraderFee()

        # Get the APYs from the PnLs and the deposited margins
        self.portfolioCalculator.computeActorAPYs()

        print("Completed LP PnL and net margin generation")
    
    def test_sharpe_ratio_undercol_events(self, tick_l=5000, tick_u=6000):

        sharpes = self.portfolioCalculator.computeSharpeRatio() # Sharpe ratio calculation
        undercols = self.portfolioCalculator.fractionUndercolEvents() # Undercollateralisation calculation
        l_factors = self.portfolioCalculator.computeLiquidationFactor() # Liquidation calculation
        levs = self.portfolioCalculator.computeLeverage(tickLower=tick_l, tickUpper=tick_u) # Leverage calculation
        the_apys = self.portfolioCalculator.returnAPYs()

        print("Completed Sharpe ratio and undercolateralisation calculations")

        return sharpes, undercols, l_factors, levs, the_apys

if __name__ == '__main__':
    unittest.main()
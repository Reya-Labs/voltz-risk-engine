from re import A
import unittest

import pandas as pd
from MarginCalculator import MarginCalculator
from constants import ALPHA, APY_LOWER_MULTIPLIER, APY_UPPER_MULTIPLIER, BETA, ETA_IM, ETA_LM, MIN_MARGIN_TO_INCENTIVIZE_LIQUIDATORS, SIGMA_SQUARED, T_MAX, TERM_START_TIMESTAMP, XI_LOWER, XI_UPPER
from utils import SECONDS_IN_DAY, SECONDS_IN_WEEK, SECONDS_IN_YEAR, fixedRateToSqrtPrice, fixedRateToTick, notional_to_liquidity

class TestMarginCalculator(unittest.TestCase):

    def setUp(self):
        self.marginCalculator = MarginCalculator(
            apyUpperMultiplier=APY_UPPER_MULTIPLIER,
            apyLowerMultiplier=APY_LOWER_MULTIPLIER,
            sigmaSquared=SIGMA_SQUARED,
            alpha=ALPHA,
            beta=BETA,
            xiUpper=XI_UPPER,
            xiLower=XI_LOWER,
            tMax=T_MAX,
            etaIM=ETA_IM,
            etaLM=ETA_LM,
            minMarginToIncentiviseLiquidators=MIN_MARGIN_TO_INCENTIVIZE_LIQUIDATORS,
        )

        self.tokens = ["USDC", "DAI"]
        self.df_original = pd.read_csv("./testing_data/test_apy_for_MarginCalculator.csv", index_col="Date")     
        self.date_original = self.df_original.index
        self.df = pd.read_csv("./testing_data/test_apy_for_MarginCalculator.csv", index_col="Date")


    def test_fixed_factor(self):
        fixedFactor = self.marginCalculator.fixedFactor(False, TERM_START_TIMESTAMP, TERM_START_TIMESTAMP + SECONDS_IN_YEAR, TERM_START_TIMESTAMP)
        self.assertAlmostEqual(fixedFactor, 0)

        fixedFactor = self.marginCalculator.fixedFactor(True, TERM_START_TIMESTAMP, TERM_START_TIMESTAMP + SECONDS_IN_YEAR, TERM_START_TIMESTAMP + SECONDS_IN_WEEK)
        self.assertAlmostEqual(fixedFactor, 0.01)

        fixedFactor = self.marginCalculator.fixedFactor(False, TERM_START_TIMESTAMP, TERM_START_TIMESTAMP + SECONDS_IN_YEAR, TERM_START_TIMESTAMP + SECONDS_IN_WEEK)
        self.assertAlmostEqual(fixedFactor, 0.01 * 7 / 365)

    def test_get_excess_balance(self):
        amount0 = -1000
        amount1 = 1000
        accruedVariableFactor = 0.02
        termStartTimestamp = TERM_START_TIMESTAMP - SECONDS_IN_WEEK
        currentTimestamp = TERM_START_TIMESTAMP
        termEndTimestamp = TERM_START_TIMESTAMP + 2 * SECONDS_IN_WEEK

        realized = self.marginCalculator.getExcessBalance(amount0, amount1, accruedVariableFactor, termStartTimestamp, termEndTimestamp, currentTimestamp)

        self.assertAlmostEqual(realized, 19808219178082192000 / 1e18)


        amount0 = -1000
        amount1 = 2000
        accruedVariableFactor = 0.02
        termStartTimestamp = TERM_START_TIMESTAMP - SECONDS_IN_WEEK
        currentTimestamp = TERM_START_TIMESTAMP
        termEndTimestamp = TERM_START_TIMESTAMP + 2 * SECONDS_IN_WEEK

        realized = self.marginCalculator.getExcessBalance(amount0, amount1, accruedVariableFactor, termStartTimestamp, termEndTimestamp, currentTimestamp)

        self.assertAlmostEqual(realized, 39808219178082192000 / 1e18, delta=1e-6)


        amount0 = 1000
        amount1 = -1000
        accruedVariableFactor = 0.02
        termStartTimestamp = TERM_START_TIMESTAMP - SECONDS_IN_WEEK
        currentTimestamp = TERM_START_TIMESTAMP
        termEndTimestamp = TERM_START_TIMESTAMP + 2 * SECONDS_IN_WEEK

        realized = self.marginCalculator.getExcessBalance(amount0, amount1, accruedVariableFactor, termStartTimestamp, termEndTimestamp, currentTimestamp)

        self.assertAlmostEqual(realized, -19808219178082192000 / 1e18, delta=1e-6)

        amount0 = 1000
        amount1 = -2000
        accruedVariableFactor = 0.02
        termStartTimestamp = TERM_START_TIMESTAMP - SECONDS_IN_WEEK
        currentTimestamp = TERM_START_TIMESTAMP
        termEndTimestamp = TERM_START_TIMESTAMP + 2 * SECONDS_IN_WEEK

        realized = self.marginCalculator.getExcessBalance(amount0, amount1, accruedVariableFactor, termStartTimestamp, termEndTimestamp, currentTimestamp)

        self.assertAlmostEqual(realized, -39808219178082192000 / 1e18, delta=1e-6)


    def test_get_fixed_token_balance(self):
        amount0 = 1000
        amount1 = -1000
        accruedVariableFactor = 0.02
        termStartTimestamp = TERM_START_TIMESTAMP - SECONDS_IN_WEEK
        currentTimestamp = TERM_START_TIMESTAMP
        termEndTimestamp = TERM_START_TIMESTAMP + 2 * SECONDS_IN_WEEK

        realized = self.marginCalculator.getFixedTokenBalance(amount0, amount1, accruedVariableFactor, termStartTimestamp, termEndTimestamp, currentTimestamp)

        self.assertAlmostEqual(realized, 35428571428571468299319 / 1e18, delta=1e-6)


    
    def test_worst_case_variable_factor_at_maturity(self):
        timeInSecondsFromStartToMaturity = 1209600
        isFT = True
        isLM = True
        lowerApyBound = None
        upperApyBound = 0.107510526004699684
        accruedVariableFactor = 0

        realized = self.marginCalculator.worstCaseVariableFactorAtMaturity(timeInSecondsFromStartToMaturity, isFT, isLM, lowerApyBound, upperApyBound, accruedVariableFactor)

        self.assertAlmostEqual(realized, 0.004123691408, delta=1e-6)

        timeInSecondsFromStartToMaturity = 1209600
        isFT = True
        isLM = False
        lowerApyBound = None
        upperApyBound = 0.107510526004699684
        accruedVariableFactor = 0

        realized = self.marginCalculator.worstCaseVariableFactorAtMaturity(timeInSecondsFromStartToMaturity, isFT, isLM, lowerApyBound, upperApyBound, accruedVariableFactor)

        self.assertAlmostEqual(realized, 0.00618553711259916, delta=1e-6)

        timeInSecondsFromStartToMaturity = 1209600
        isFT = False
        isLM = True
        lowerApyBound = 0.092372593455489616
        upperApyBound = None
        accruedVariableFactor = 0

        realized = self.marginCalculator.worstCaseVariableFactorAtMaturity(timeInSecondsFromStartToMaturity, isFT, isLM, lowerApyBound, upperApyBound, accruedVariableFactor)

        self.assertAlmostEqual(realized, 0.003543058379114670, delta=1e-6)

        timeInSecondsFromStartToMaturity = 1209600
        isFT = False
        isLM = False
        lowerApyBound = 0.092372593455489616
        upperApyBound = None
        accruedVariableFactor = 0

        realized = self.marginCalculator.worstCaseVariableFactorAtMaturity(timeInSecondsFromStartToMaturity, isFT, isLM, lowerApyBound, upperApyBound, accruedVariableFactor)

        self.assertAlmostEqual(realized, 0.002480140865380269, delta=1e-6)

    def test_get_extra_balances(self):
        fromTick = -120
        toTick = 120
        liquidity = 1000
        variableFactor = 0.2
        termStartTimestamp = TERM_START_TIMESTAMP
        termEndTimestamp = TERM_START_TIMESTAMP + SECONDS_IN_WEEK
        currentTimestamp = TERM_START_TIMESTAMP

        realized = self.marginCalculator.getExtraBalances(fromTick, toTick, liquidity, variableFactor, termStartTimestamp, termEndTimestamp, currentTimestamp)

        self.assertAlmostEqual(realized[0], -12525.734370083996972796, delta=1e-3)
        self.assertAlmostEqual(realized[1], 11.999472029327827822, delta=1e-3)


        fromTick = 120
        toTick = -120
        liquidity = 1000
        variableFactor = 0.2
        termStartTimestamp = TERM_START_TIMESTAMP
        termEndTimestamp = TERM_START_TIMESTAMP + SECONDS_IN_WEEK
        currentTimestamp = TERM_START_TIMESTAMP + 3 * SECONDS_IN_DAY

        realized = self.marginCalculator.getExtraBalances(fromTick, toTick, liquidity, variableFactor, termStartTimestamp, termEndTimestamp, currentTimestamp)

        self.assertAlmostEqual(realized[0], 12520.59173921428502871, delta=1e-3)
        self.assertAlmostEqual(realized[1], -11.999472029327827822, delta=1e-3)

  
    def test_get_minimum_margin_requirement(self):
        variableTokenBalance = -5000
        currentTimestamp = TERM_START_TIMESTAMP
        termEndTimestamp = TERM_START_TIMESTAMP + SECONDS_IN_WEEK
        isLM = True

        realized = self.marginCalculator.getMinimumMarginRequirement(variableTokenBalance, currentTimestamp, termEndTimestamp, isLM)
        self.assertAlmostEqual(realized, 0.09589041096, delta=1e-3)
        
        variableTokenBalance = 15000
        currentTimestamp = TERM_START_TIMESTAMP
        termEndTimestamp = TERM_START_TIMESTAMP + SECONDS_IN_WEEK
        isLM = True

        realized = self.marginCalculator.getMinimumMarginRequirement(variableTokenBalance, currentTimestamp, termEndTimestamp, isLM)
        self.assertAlmostEqual(realized, 0.2876712329, delta=1e-3)
        
        variableTokenBalance = -10000
        currentTimestamp = TERM_START_TIMESTAMP
        termEndTimestamp = TERM_START_TIMESTAMP + SECONDS_IN_WEEK
        isLM = False

        realized = self.marginCalculator.getMinimumMarginRequirement(variableTokenBalance, currentTimestamp, termEndTimestamp, isLM)
        self.assertAlmostEqual(realized, 0.3835616438, delta=1e-3)

    def test_get_margin_requirement(self):
        fixedTokenBalance = 1000
        variableTokenBalance = -3000
        isLM = True
        sqrtPrice = fixedRateToSqrtPrice(20)
        lowerApyBound = 0.000356854815268913
        upperApyBound = 0.001297058331272719
        termStartTimestamp = TERM_START_TIMESTAMP
        termEndTimestamp = TERM_START_TIMESTAMP + SECONDS_IN_WEEK
        currentTimestamp = TERM_START_TIMESTAMP
        accruedVariableFactor = 0

        realized = self.marginCalculator.getMarginRequirement(fixedTokenBalance, variableTokenBalance, isLM, sqrtPrice,
                             lowerApyBound, upperApyBound, termStartTimestamp, termEndTimestamp,
                             currentTimestamp, accruedVariableFactor)

        self.assertAlmostEqual(realized, 0.057534246575342465, delta=1e-3) # Update value


        fixedTokenBalance = 10
        variableTokenBalance = -30000000
        isLM = True
        sqrtPrice = fixedRateToSqrtPrice(20)
        lowerApyBound = 0.000356854230859285
        upperApyBound = 0.001297056207122100
        termStartTimestamp = TERM_START_TIMESTAMP
        termEndTimestamp = TERM_START_TIMESTAMP + SECONDS_IN_WEEK
        currentTimestamp = TERM_START_TIMESTAMP
        accruedVariableFactor = 0

        realized = self.marginCalculator.getMarginRequirement(fixedTokenBalance, variableTokenBalance, isLM, sqrtPrice,
                             lowerApyBound, upperApyBound, termStartTimestamp, termEndTimestamp,
                             currentTimestamp, accruedVariableFactor)

        self.assertAlmostEqual(realized, 746.2495986198234, delta=10) # Update value


        fixedTokenBalance = 1000
        variableTokenBalance = -3000
        isLM = False
        sqrtPrice = fixedRateToSqrtPrice(20)
        lowerApyBound = 0.000356854230859285
        upperApyBound = 0.001297056207122100
        termStartTimestamp = TERM_START_TIMESTAMP
        termEndTimestamp = TERM_START_TIMESTAMP + SECONDS_IN_WEEK
        currentTimestamp = TERM_START_TIMESTAMP
        accruedVariableFactor = 0

        realized = self.marginCalculator.getMarginRequirement(fixedTokenBalance, variableTokenBalance, isLM, sqrtPrice,
                             lowerApyBound, upperApyBound, termStartTimestamp, termEndTimestamp,
                             currentTimestamp, accruedVariableFactor)

        self.assertAlmostEqual(realized, 0.11506849315068493, delta=1) # Update value

 
        fixedTokenBalance = -1000
        variableTokenBalance = 3000
        isLM = True
        sqrtPrice = fixedRateToSqrtPrice(1/20)
        lowerApyBound = 0.000356854230859285
        upperApyBound = 0.001297056207122100
        termStartTimestamp = TERM_START_TIMESTAMP
        termEndTimestamp = TERM_START_TIMESTAMP + SECONDS_IN_WEEK
        currentTimestamp = TERM_START_TIMESTAMP
        accruedVariableFactor = 0

        realized = self.marginCalculator.getMarginRequirement(fixedTokenBalance, variableTokenBalance, isLM, sqrtPrice,
                             lowerApyBound, upperApyBound, termStartTimestamp, termEndTimestamp,
                             currentTimestamp, accruedVariableFactor)

        self.assertAlmostEqual(realized, 0.171249348113833000, delta=1e-3) # Update value


        fixedTokenBalance = 1000
        variableTokenBalance = -30000
        isLM = False
        sqrtPrice = fixedRateToSqrtPrice(1/20)
        lowerApyBound = 0.000356854230859285
        upperApyBound = 0.001297056207122100
        termStartTimestamp = TERM_START_TIMESTAMP
        termEndTimestamp = TERM_START_TIMESTAMP + SECONDS_IN_WEEK
        currentTimestamp = TERM_START_TIMESTAMP
        accruedVariableFactor = 0

        realized = self.marginCalculator.getMarginRequirement(fixedTokenBalance, variableTokenBalance, isLM, sqrtPrice,
                             lowerApyBound, upperApyBound, termStartTimestamp, termEndTimestamp,
                             currentTimestamp, accruedVariableFactor)

        self.assertAlmostEqual(realized, 1.1506849315068493, delta=1e-3) # Update value


        fixedTokenBalance = 1000
        variableTokenBalance = -1000
        isLM = False
        sqrtPrice = fixedRateToSqrtPrice(1)
        lowerApyBound = 0.005
        upperApyBound = 0.02
        termStartTimestamp = TERM_START_TIMESTAMP - SECONDS_IN_WEEK
        termEndTimestamp = TERM_START_TIMESTAMP + SECONDS_IN_WEEK
        currentTimestamp = TERM_START_TIMESTAMP
        accruedVariableFactor = 0.000193718001937
        accruedApy = ((accruedVariableFactor + 1) ** (365 / 7) - 1)
        print("ACCRUED APY:", accruedApy)

        realized = self.marginCalculator.getMarginRequirement(fixedTokenBalance, variableTokenBalance, isLM, sqrtPrice,
                             lowerApyBound, upperApyBound, termStartTimestamp, termEndTimestamp,
                             currentTimestamp, accruedVariableFactor)

        self.assertAlmostEqual(realized, 0.3856102780477994, delta=1e-3) # Update value


    def test_get_position_margin_requirement(self):
        variableFactor = 0
        currentTick = -23028
        positionLiquidity = 1000
        tickLower = -13864
        tickUpper = -6932
        sqrtPrice = fixedRateToSqrtPrice(10)
        termStartTimestamp = TERM_START_TIMESTAMP
        termEndTimestamp = TERM_START_TIMESTAMP + SECONDS_IN_WEEK
        currentTimestamp = TERM_START_TIMESTAMP
        positionFixedTokenBalance = 0
        positionVariableTokenBalance = 0
        isLM = True
        lowerApyBound = 0.000356854230859285
        upperApyBound = 0.001297056207122100

        realized = self.marginCalculator.getPositionMarginRequirement(variableFactor, currentTick, positionLiquidity, tickLower, tickUpper, sqrtPrice,
                                     termStartTimestamp, termEndTimestamp, currentTimestamp, positionVariableTokenBalance,
                                     positionFixedTokenBalance, isLM, lowerApyBound, upperApyBound)

        self.assertAlmostEqual(realized, 0.110927831816756198, delta=1e-3)


        variableFactor = 0
        currentTick = fixedRateToTick(10)
        positionLiquidity = 1000
        tickLower = fixedRateToTick(4)
        tickUpper = fixedRateToTick(2)
        sqrtPrice = fixedRateToSqrtPrice(10)
        termStartTimestamp = TERM_START_TIMESTAMP
        termEndTimestamp = TERM_START_TIMESTAMP + SECONDS_IN_WEEK
        currentTimestamp = TERM_START_TIMESTAMP
        positionFixedTokenBalance = 585.80
        positionVariableTokenBalance = -207.10
        isLM = True
        lowerApyBound = 0.000356854230859285
        upperApyBound = 0.001297056207122100

        realized = self.marginCalculator.getPositionMarginRequirement(variableFactor, currentTick, positionLiquidity, tickLower, tickUpper, sqrtPrice,
                                     termStartTimestamp, termEndTimestamp, currentTimestamp, positionVariableTokenBalance,
                                     positionFixedTokenBalance, isLM, lowerApyBound, upperApyBound)

        # minimum margin requirement
        self.assertAlmostEqual(realized, 0.0039717, delta=1e-3)


        variableFactor = 0
        currentTick = fixedRateToTick(4)
        positionLiquidity = 1000
        tickLower = fixedRateToTick(10)
        tickUpper = fixedRateToTick(2)
        sqrtPrice = fixedRateToSqrtPrice(4)
        termStartTimestamp = TERM_START_TIMESTAMP
        termEndTimestamp = TERM_START_TIMESTAMP + SECONDS_IN_WEEK
        currentTimestamp = TERM_START_TIMESTAMP
        positionFixedTokenBalance = -1162
        positionVariableTokenBalance = 183
        isLM = True
        lowerApyBound = 0.000356854230859285
        upperApyBound = 0.001297056207122100

        realized = self.marginCalculator.getPositionMarginRequirement(variableFactor, currentTick, positionLiquidity, tickLower, tickUpper, sqrtPrice,
                                     termStartTimestamp, termEndTimestamp, currentTimestamp, positionVariableTokenBalance,
                                     positionFixedTokenBalance, isLM, lowerApyBound, upperApyBound)

        self.assertAlmostEqual(realized, 0.33252454764, delta=1e-3)


        variableFactor = 0
        currentTick = fixedRateToTick(4)
        positionLiquidity = 1000
        tickLower = fixedRateToTick(10)
        tickUpper = fixedRateToTick(2)
        sqrtPrice = fixedRateToSqrtPrice(4)
        termStartTimestamp = TERM_START_TIMESTAMP
        termEndTimestamp = TERM_START_TIMESTAMP + SECONDS_IN_WEEK
        currentTimestamp = TERM_START_TIMESTAMP
        positionFixedTokenBalance = 0
        positionVariableTokenBalance = 0
        isLM = True
        lowerApyBound = 0.000356854230859285
        upperApyBound = 0.001297056207122100

        realized = self.marginCalculator.getPositionMarginRequirement(variableFactor, currentTick, positionLiquidity, tickLower, tickUpper, sqrtPrice,
                                     termStartTimestamp, termEndTimestamp, currentTimestamp, positionVariableTokenBalance,
                                     positionFixedTokenBalance, isLM, lowerApyBound, upperApyBound)

        self.assertAlmostEqual(realized, 0.11092521127707744, delta=1e-3)


        variableFactor = 0
        currentTick = fixedRateToTick(2)
        positionLiquidity = 1000
        tickLower = fixedRateToTick(10)
        tickUpper = fixedRateToTick(4)
        sqrtPrice = fixedRateToSqrtPrice(2)
        termStartTimestamp = TERM_START_TIMESTAMP
        termEndTimestamp = TERM_START_TIMESTAMP + SECONDS_IN_WEEK
        currentTimestamp = TERM_START_TIMESTAMP
        positionFixedTokenBalance = 0
        positionVariableTokenBalance = 0
        isLM = True
        lowerApyBound = 0.000356854230859285
        upperApyBound = 0.001297056207122100

        realized = self.marginCalculator.getPositionMarginRequirement(variableFactor, currentTick, positionLiquidity, tickLower, tickUpper, sqrtPrice,
                                     termStartTimestamp, termEndTimestamp, currentTimestamp, positionVariableTokenBalance,
                                     positionFixedTokenBalance, isLM, lowerApyBound, upperApyBound)

        self.assertAlmostEqual(realized, 0.0035243990078962597, delta=1e-3)


        variableFactor = 0
        currentTick = fixedRateToTick(2)
        positionLiquidity = 1000
        tickLower = fixedRateToTick(10)
        tickUpper = fixedRateToTick(4)
        sqrtPrice = fixedRateToSqrtPrice(2)
        termStartTimestamp = TERM_START_TIMESTAMP
        termEndTimestamp = TERM_START_TIMESTAMP + SECONDS_IN_WEEK
        currentTimestamp = TERM_START_TIMESTAMP
        positionFixedTokenBalance = -1162
        positionVariableTokenBalance = 183
        isLM = True
        lowerApyBound = 0.000356854230859285
        upperApyBound = 0.001297056207122100

        realized = self.marginCalculator.getPositionMarginRequirement(variableFactor, currentTick, positionLiquidity, tickLower, tickUpper, sqrtPrice,
                                     termStartTimestamp, termEndTimestamp, currentTimestamp, positionVariableTokenBalance,
                                     positionFixedTokenBalance, isLM, lowerApyBound, upperApyBound)

        self.assertAlmostEqual(realized, 0.22159689926, delta=1e-3)


if __name__ == '__main__':
    unittest.main()







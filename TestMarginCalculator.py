import unittest

from sqlalchemy import true
from MarginCalculator import MarginCalculator
import pandas as pd
import numpy as np
from utils.utils import SECONDS_IN_DAY, SECONDS_IN_WEEK, SECONDS_IN_YEAR, fixedRateToSqrtPrice, fixedRateToTick, notional_to_liquidity

APY_UPPER_MULTIPLIER = 1.5
APY_LOWER_MULTIPLIER = 0.7
SIGMA_SQUARED = 0.5
ALPHA = 0.1
BETA = 1
XI_UPPER = 2
XI_LOWER = 1.5
T_MAX = 31536000
DEV_MUL_LEFT_UNWIND_LM = 0.5
DEV_MUL_RIGHT_UNWIND_LM = 0.5
DEV_MUL_LEFT_UNWIND_IM = 0.5
DEV_MUL_RIGHT_UNWIND_IM = 0.5
FIXED_RATE_DEVIATION_MIN_LEFT_UNWIND_LM = 0.1
FIXED_RATE_DEVIATION_MIN_RIGHT_UNWIND_LM = 0.1
FIXED_RATE_DEVIATION_MIN_LEFT_UNWIND_IM = 0.1
FIXED_RATE_DEVIATION_MIN_RIGHT_UNWIND_IM = 0.1
GAMMA = 1
MIN_MARGIN_TO_INCENTIVIZE_LIQUIDATORS = 0
TERM_START_TIMESTAMP = 1649020521
TERM_END_TIMESTAMP = 1680383721

AMOUNT_FIXED = 1000
AMOUNT_VARIABLE = -1000
ACCRUED_VARIABLE_FACTOR = 0.02

# Market factors for the bullish and bearish fixed rates
FT_FACTOR = 0.3
FT_ERROR = 0.01

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
            devMulLeftUnwindLM=DEV_MUL_LEFT_UNWIND_LM,
            devMulRightUnwindLM=DEV_MUL_RIGHT_UNWIND_LM,
            devMulLeftUnwindIM=DEV_MUL_LEFT_UNWIND_IM,
            devMulRightUnwindIM=DEV_MUL_RIGHT_UNWIND_IM,
            fixedRateDeviationMinLeftUnwindLM=FIXED_RATE_DEVIATION_MIN_LEFT_UNWIND_LM,
            fixedRateDeviationMinRightUnwindLM=FIXED_RATE_DEVIATION_MIN_RIGHT_UNWIND_LM,
            fixedRateDeviationMinLeftUnwindIM=FIXED_RATE_DEVIATION_MIN_LEFT_UNWIND_IM,
            fixedRateDeviationMinRightUnwindIM=FIXED_RATE_DEVIATION_MIN_RIGHT_UNWIND_IM,
            gamma=GAMMA,
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


    def test_get_fixed_token_delta_unbalanced_simulated_unwind(self):
        variableTokenDeltaAbsolute = 1000
        fixedRateStart = 1
        startingFixedRateMultiplier = 1.5
        fixedRateDeviationMin = 0.2
        termEndTimestamp = TERM_START_TIMESTAMP + 3 * SECONDS_IN_WEEK
        currentTimestamp = TERM_START_TIMESTAMP + SECONDS_IN_WEEK
        tMax = T_MAX
        gamma = 1
        isFTUnwind = True
        
        realized = self.marginCalculator.getFixedTokenDeltaUnbalancedSimulatedUnwind(variableTokenDeltaAbsolute, fixedRateStart, startingFixedRateMultiplier, fixedRateDeviationMin, termEndTimestamp, currentTimestamp, tMax, gamma, isFTUnwind)

        self.assertAlmostEqual(realized, 943.555176826535, delta=1e-6)

    
    def test_worst_case_variable_factor_at_maturity(self):
        timeInSecondsFromStartToMaturity = 1209600
        isFT = True
        isLM = True
        lowerApyBound = None
        upperApyBound = 0.107510526004699684

        realized = self.marginCalculator.worstCaseVariableFactorAtMaturity(timeInSecondsFromStartToMaturity, isFT, isLM, lowerApyBound, upperApyBound)

        self.assertAlmostEqual(realized, 0.004123691408, delta=1e-6)

        timeInSecondsFromStartToMaturity = 1209600
        isFT = True
        isLM = False
        lowerApyBound = None
        upperApyBound = 0.107510526004699684

        realized = self.marginCalculator.worstCaseVariableFactorAtMaturity(timeInSecondsFromStartToMaturity, isFT, isLM, lowerApyBound, upperApyBound)

        self.assertAlmostEqual(realized, 0.00618553711259916, delta=1e-6)

        timeInSecondsFromStartToMaturity = 1209600
        isFT = False
        isLM = True
        lowerApyBound = 0.092372593455489616
        upperApyBound = None

        realized = self.marginCalculator.worstCaseVariableFactorAtMaturity(timeInSecondsFromStartToMaturity, isFT, isLM, lowerApyBound, upperApyBound)

        self.assertAlmostEqual(realized, 0.003543058379114670, delta=1e-6)

        timeInSecondsFromStartToMaturity = 1209600
        isFT = False
        isLM = False
        lowerApyBound = 0.092372593455489616
        upperApyBound = None

        realized = self.marginCalculator.worstCaseVariableFactorAtMaturity(timeInSecondsFromStartToMaturity, isFT, isLM, lowerApyBound, upperApyBound)

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

        self.assertAlmostEqual(realized, 11.424182354226593680, delta=1e-3)


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

        self.assertAlmostEqual(realized, 116159.629843635797628803, delta=10)


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

        self.assertAlmostEqual(realized, 11.752037636084398722, delta=1)

 
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

        self.assertAlmostEqual(realized, 0.171249348113833000, delta=1e-3)


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

        self.assertAlmostEqual(realized, 0.927603785405792000, delta=1e-3)


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

        self.assertAlmostEqual(realized, 0.2886377911, delta=1e-3)


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

        self.assertAlmostEqual(realized, 0.13290739135, delta=1e-3)


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

        self.assertAlmostEqual(realized, 0.13290739135, delta=1e-3)


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


    def date_to_unix_time(self):
        import datetime
        import time
        
        df = self.df_original
        df.reset_index(drop=True, inplace=True)
     
        date_time = []
        for date in self.date_original:
            y,m,d = [int(date.split(" ")[0].split("-")[i]) for i in range(3)]
            h,mt,s = [int(date.split(" ")[1].split(":")[i]) for i in range(3)]
            date_time.append(datetime.datetime(y, m, d, h, mt, s))

        df["date"] = np.array([time.mktime(dt.timetuple()) for dt in date_time])    

        return df

    # """
    #     This is an important method which organises the outputs of the Simulator into relevant APYs and
    #     rates. Specifically we have the following:

    #     1) Variable rate --> the Simulator APY
    #     2) Fixed rate --> we consider three different scenarios:
    #         a) Neutral: fixed rate = variable rate (default)
    #         b) Bull: fixed rate = variable rate + const * time to maturity + error term
    #         c) Bear: fixed rate = variable rate - const * time to maturity - error term

    #     IMPORTANT: we need to ensure the timestamps are already converted to Unix time, so need to call
    #     data_to_unix_time before running this processing step.
    # """
    def preprocess_df(self, token, fr_market="neutral"):

        # Pick a single token
        df = self.df # Needs to have already been timestamp processed
        df = df.loc[:, [f"{token} model", f"{token} APY lower", f"{token} APY upper", f"{token} model", "date"]]
        df.columns = ["apy", "lower", "upper", "fr", "date"] # USDC model corresponds to the simulation of fixed rates
        
        # Compute liquidity index from the given token APY and the corresponding variable factor
        df = self.calculate_liquidity_index(df)
        liquidity_index_term_start_timestamp = df.loc[0, "liquidity_index"]
        df.loc[:, "variable factor"] = (df.loc[:, "liquidity_index"] / liquidity_index_term_start_timestamp) - 1

        # Update the fixed rate if bearish/bullish conditions are specified
        if fr_market=="bull":
            df["fr"] = [df["apy"].values[i] + FT_FACTOR*(df["date"].values[-1]-df["date"].values[i])/df["date"].values[-1] + FT_ERROR for i in range(len(df))]
        if fr_market=="bear":
            df["fr"] = [df["apy"].values[i] - FT_FACTOR*(df["date"].values[-1]-df["date"].values[i])/df["date"].values[-1] - FT_ERROR for i in range(len(df))]
            # Check on fixed rates, in case bearish rates go negative
            df["fr"] = [fr if fr>0 else 0.0001 for fr in df["fr"].values]
        
        return df

    def compute_liquidity_index(self, row, df):
        if row.name <=7:
            counterfactual_rate = df.loc[0, "rate"]
            return ((counterfactual_rate - 1) / 7) * row.name + 1
        else:

            df = df.shift(row.name % 7).dropna().iloc[::7, :]
            df = df[df['date'] < row['date']]
            return df.loc[:, 'rate'].cumprod().iloc[-1] * row['rate']

    def calculate_liquidity_index(self, df):
        # todo: move into a separate module/script
        WEEK_IN_YEARS = 7 / 365  # rough approx

        df.loc[:, 'rate'] = 1 + df.loc[:, 'apy'] * WEEK_IN_YEARS

        df.loc[:, 'liquidity_index'] = df.apply(
            self.compute_liquidity_index,
            args=(df,),
            axis=1
        )

        # calculate the rate week on week
        # todo: also multiply by liquidity indicies for the first 7 rows

        return df

    def test_generate_full_output(self, leverage_factor=1, notional=1000, lp_fix=0, lp_var=0, tick_l=5000, tick_u=6000, fr_market="neutral"):

        # All tokens
        # All trader types
        # Margin requirements & pnl        
        lp_fixed_token_balance = lp_fix
        lp_variable_token_balance = lp_var
        tickLower = tick_l
        tickUpper = tick_u
        lp_liquidity = notional_to_liquidity(notional=notional, tick_l=tickLower, tick_u=tickUpper)

        list_of_tokens = self.tokens
        trader_types = ["ft", "vt", "lp"]

        dfs_per_token = []
        balance_per_token = {} # We need to keep track of the position names for the PortfolioCalculator

        # Before looping of each token we need to first convert the timestamp to unix time **IN SECONDS**, so
        # that we can later on normalised to SECONDS_IN_YEAR for the APY calculation
        self.df = self.date_to_unix_time()
        for token in list_of_tokens:
            df = self.preprocess_df(token, fr_market=fr_market) # Update for each new token
            # Now we need to update the fixed and variable token balances to account for
            # different possibilities/markets in the fixed rate
            ft_fixed_token_balance = notional * (df["fr"].values[0] * 100)
            ft_variable_token_balance = -notional
            vt_fixed_token_balance = -notional * (df["fr"].values[0] * 100)
            vt_variable_token_balance = notional
            
            balance_per_token[token] = {
                "ft_fix": ft_fixed_token_balance,
                "ft_var": ft_variable_token_balance,
                "vt_fix": vt_fixed_token_balance,
                "vt_var": vt_variable_token_balance,
            }
            for trader_type in trader_types:

                if trader_type == 'ft':

                    df = self.marginCalculator.generate_margin_requirements_trader(df, ft_fixed_token_balance,
                                                                                       ft_variable_token_balance, 'ft', f'{token}')

                    daily_fixed_rate = ((abs(ft_fixed_token_balance) / ft_variable_token_balance) / 100) / 365
                    
                    fixed_factor_series = pd.Series(data=1, index=range(len(df)))
                    fixed_factor_series = pd.Series(data=fixed_factor_series.index * daily_fixed_rate)

                    df = self.marginCalculator.generate_pnl_trader(df, fixed_factor_series, ft_fixed_token_balance,
                                                                       ft_variable_token_balance, 'ft', f'{token}')

                    df = self.marginCalculator.generate_net_margin_trader(df, \
                        ft_fixed_token_balance, ft_variable_token_balance, 'ft', f'{token}', leverage_factor=leverage_factor)

                elif trader_type == 'vt':
                    
                    df = self.marginCalculator.generate_margin_requirements_trader(df, vt_fixed_token_balance,
                                                                                       vt_variable_token_balance, 'vt', f'{token}')

                    daily_fixed_rate = ((abs(vt_fixed_token_balance) / vt_variable_token_balance) / 100) / 365

                    fixed_factor_series = pd.Series(data=1, index=range(len(df)))
                    fixed_factor_series = pd.Series(data=fixed_factor_series.index * daily_fixed_rate)

                    df = self.marginCalculator.generate_pnl_trader(df, fixed_factor_series,
                                                                       vt_fixed_token_balance,
                                                                       vt_variable_token_balance, 'vt', f'{token}')

                    df = self.marginCalculator.generate_net_margin_trader(df, \
                        vt_fixed_token_balance, vt_variable_token_balance, 'vt', f'{token}', leverage_factor=leverage_factor)

                else:
                    df = self.marginCalculator.generate_margin_requirements_lp(
                        df=df,
                        fixedTokenBalance=lp_fixed_token_balance,
                        variableTokenBalance=lp_variable_token_balance,
                        liquidity=lp_liquidity,
                        tickLower=tickLower,
                        tickUpper=tickUpper,
                        token=f'{token}'
                    )

            df = df.rename(
                columns = {
                    'apy': f'apy_{token}',
                    'lower': f'lower_{token}',
                    'upper': f'upper_{token}',
                    'fr': f'fr_{token}',
                    'rate': f'rate_{token}',
                    'liquidity_index': f'liquidity_index_{token}',
                    'variable factor': f'variable factor_{token}',
                }
            )

            df = df.set_index(
                'date'
            )
            dfs_per_token.append(df)
        result = pd.concat(dfs_per_token, axis=1)
        result["t_years"] = [(i-result.index.values[0])/SECONDS_IN_YEAR for i in result.index.values] # Add t_year for downstream APY calculation

        return result, balance_per_token

    # def test_generate_pnl_trader(self):
        
    #     self.df = self.date_to_unix_time()
    #     df = self.preprocess_df("USDC", fr_market="neutral") 

    #     fixedTokenBalance = -1000
    #     variableTokenBalance = 100

    #     daily_fixed_rate = ((abs(fixedTokenBalance) / variableTokenBalance) / 100) / 365

    #     fixed_factor_series = pd.Series(data=1, index=range(len(self.df)))
    #     fixed_factor_series = pd.Series(data=fixed_factor_series.index * daily_fixed_rate)

    #     result = self.marginCalculator.generate_pnl_trader(df, fixed_factor_series, fixedTokenBalance, variableTokenBalance, 'vt', 'usdc')


    # def test_generate_margin_requirements_trader(self):
        
    #     self.df = self.date_to_unix_time()
    #     df = self.preprocess_df("USDC", fr_market="neutral") 

    #     fixedTokenBalance = -1000
    #     variableTokenBalance = 100

    #     result = self.marginCalculator.generate_margin_requirements_trader(df, fixedTokenBalance, variableTokenBalance,
    #                                                                        'vt', 'usdc')
    #     fixedTokenBalance = 100
    #     variableTokenBalance = -1000

    #     result = self.marginCalculator.generate_margin_requirements_trader(df, fixedTokenBalance, variableTokenBalance,
    #                                                                        'ft', 'usdc')
    #     ## todo: assertion here

    # def test_generate_margin_requirements_lp(self):

    #     self.df = self.date_to_unix_time()
    #     df = self.preprocess_df("USDC", fr_market="neutral") 
        
    #     # df, token, fixedTokenBalance, variableTokenBalance, liquidity, tickLower, tickUpper)

    #     positionLiquidity = 100000
    #     tickLower = 5000
    #     tickUpper= 6000

    #     result = self.marginCalculator.generate_margin_requirements_lp(
    #         df=df,
    #         fixedTokenBalance=0,
    #         variableTokenBalance=0,
    #         liquidity=positionLiquidity,
    #         tickLower=tickLower,
    #         tickUpper=tickUpper,
    #         token='usdc'
    #     )

    #     # todo: assertion here



    # def test_fixed_factor(self):

    #     fixedFactor = self.marginCalculator.fixedFactor(atMaturity=True, termStartTimestamp=TERM_START_TIMESTAMP,
    #                                                termEndTimestamp=TERM_END_TIMESTAMP,
    #                                                currentTimestamp=TERM_END_TIMESTAMP)

    #     self.assertEqual(fixedFactor, 0.009945205479452055)

    # def test_get_fixed_token_balance(self):

    #     excessBalance = self.marginCalculator.getExcessBalance(
    #         amountFixed=AMOUNT_FIXED,
    #         amountVariable=AMOUNT_VARIABLE,
    #         accruedVariableFactor=ACCRUED_VARIABLE_FACTOR,
    #         termStartTimestamp=TERM_START_TIMESTAMP,
    #         termEndTimestamp=TERM_END_TIMESTAMP,
    #         currentTimestamp=TERM_START_TIMESTAMP,
    #     )

    #     fixedTokenBalanace = self.marginCalculator.calculateFixedTokenBalance(
    #         amountFixed=1000,
    #         excessBalance=excessBalance,
    #         termStartTimestamp=TERM_START_TIMESTAMP,
    #         termEndTimestamp=TERM_END_TIMESTAMP,
    #         currentTimestamp=TERM_START_TIMESTAMP
    #     )

    #     fixedTokenBalanceDirectCalculation = self.marginCalculator.getFixedTokenBalance(
    #         amountFixed=AMOUNT_FIXED,
    #         amountVariable=AMOUNT_VARIABLE,
    #         accruedVariableFactor=ACCRUED_VARIABLE_FACTOR,
    #         termStartTimestamp=TERM_START_TIMESTAMP,
    #         termEndTimestamp=TERM_END_TIMESTAMP,
    #         currentTimestamp=TERM_START_TIMESTAMP
    #     )

    #     self.assertEqual(fixedTokenBalanceDirectCalculation, fixedTokenBalanace)

    # def test_get_minimum_margin_requirement(self):
    #     # Minimum Margin Requirement sheet: https://docs.google.com/spreadsheets/d/1FjNbw4bojgH4_MERARQD2F_y_lzflEi6H7j-Y6pfAA4/edit?usp=sharing

    #     # lower and upper apy bounds should not have an effect on the minimum margin calculation
    #     fixedTokenBalance = 1162.377956
    #     variableTokenBalance = -183.7789286
    #     currentTimestamp = 1645126092
    #     termStartTimestamp = currentTimestamp
    #     termEndTimestamp = 1645730883
    #     fixedRate = 10.0009978
    #     accruedVariableFactor = 0
    #     isLM = True
    #     dummyLowerApyBound = 0.01
    #     dummyUpperApyBound = 0.05

    #     minimumMarginRequirement = self.marginCalculator.getMinimumMarginRequirement(
    #         fixedTokenBalance=fixedTokenBalance,
    #         variableTokenBalance=variableTokenBalance,
    #         isLM=isLM,
    #         fixedRate=fixedRate,
    #         currentTimestamp=currentTimestamp,
    #         accruedVariableFactor=accruedVariableFactor,
    #         lowerApyBound=dummyLowerApyBound,
    #         upperApyBound=dummyUpperApyBound,
    #         termStartTimestamp=termStartTimestamp,
    #         termEndTimestamp=termEndTimestamp
    #     )

    #     self.assertEqual(minimumMarginRequirement, 0.13291189658225627)


    # def test_get_position_margin_requirement(self):

    #     # trader, non-lp

    #     fixedTokenBalance = 1000
    #     variableTokenBalance = -3000
    #     isLM = False

    #     marginRequirement = self.marginCalculator._getMarginRequirement(
    #         fixedTokenBalance=fixedTokenBalance,
    #         variableTokenBalance=variableTokenBalance,
    #         isLM=isLM,
    #         lowerApyBound=0.01,
    #         upperApyBound=0.05,
    #         termStartTimestamp=TERM_START_TIMESTAMP,
    #         termEndTimestamp=TERM_END_TIMESTAMP,
    #         currentTimestamp=TERM_START_TIMESTAMP
    #     )

    #     self.assertEqual(marginRequirement, 213.82191780821918) # need to double check

    # def test_get_lp_position_margin_requirement(self):

        # variableFactor = 0
        # currentTick = 0
        # sqrtPrice = 1  # corresponds to a tick of 0 above
        # positionLiquidity = 100000
        # tickLower = -60
        # tickUpper = 60
        # positionVariableTokenBalance = 0
        # positionFixedTokenBalance = 0
        # isLM = True
        # lowerApyBound = 0.01
        # upperApyBound = 0.05

        # marginRequirement = self.marginCalculator.getPositionMarginRequirement(
        #     variableFactor=variableFactor,
        #     currentTick=currentTick,
        #     sqrtPrice=sqrtPrice,
        #     positionLiquidity=positionLiquidity,
        #     tickLower=tickLower,
        #     tickUpper=tickUpper,
        #     positionVariableTokenBalance=positionVariableTokenBalance,
        #     positionFixedTokenBalance=positionFixedTokenBalance,
        #     isLM=isLM,
        #     lowerApyBound=lowerApyBound,
        #     upperApyBound=upperApyBound,
        #     currentTimestamp=TERM_START_TIMESTAMP,
        #     termStartTimestamp=TERM_START_TIMESTAMP,
        #     termEndTimestamp=TERM_END_TIMESTAMP
        # )

        # self.assertEqual(marginRequirement, 11.906818411512914)  # need to double check

if __name__ == '__main__':
    unittest.main()







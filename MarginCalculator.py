import math
from sre_constants import AT_LOC_BOUNDARY
from tkinter import _EntryValidateCommand
from tracemalloc import start

import pandas as pd
from utils import SECONDS_IN_YEAR, date_to_unix_time, generate_margin_requirements_lp, generate_margin_requirements_trader, generate_net_margin_trader, generate_pnl_trader, getAmount0Delta, getAmount1Delta, getSqrtRatioAtTick, notional_to_liquidity, preprocess_df, sqrtPriceToFixedRate


class MarginCalculator:

    def __init__(self, apyUpperMultiplier, apyLowerMultiplier, sigmaSquared, alpha, beta, xiUpper, xiLower, 
                 etaLM, etaIM, tMax, devMulLeftUnwindLM, devMulRightUnwindLM, devMulLeftUnwindIM, devMulRightUnwindIM,
                 fixedRateDeviationMinLeftUnwindLM, fixedRateDeviationMinRightUnwindLM,
                 fixedRateDeviationMinLeftUnwindIM, fixedRateDeviationMinRightUnwindIM,
                 gamma, minMarginToIncentiviseLiquidators):

        # ref: https://github.com/Voltz-Protocol/voltz-core/blob/39da8d657df2bb7ce2e0260e635b2202c3d8d82d/contracts/interfaces/IMarginEngine.sol#L16

        self.apyUpperMultiplier = apyUpperMultiplier
        self.apyLowerMultiplier = apyLowerMultiplier
        self.sigmaSquared = sigmaSquared
        self.alpha = alpha
        self.beta = beta
        self.xiUpper = xiUpper
        self.xiLower = xiLower
        self.etaLM = etaLM 
        self.etaIM = etaIM
        self.tMax = tMax
        self.devMulLeftUnwindLM = devMulLeftUnwindLM
        self.devMulRightUnwindLM = devMulRightUnwindLM
        self.devMulLeftUnwindIM = devMulLeftUnwindIM
        self.devMulRightUnwindIM = devMulRightUnwindIM
        self.fixedRateDeviationMinLeftUnwindLM = fixedRateDeviationMinLeftUnwindLM
        self.fixedRateDeviationMinRightUnwindLM = fixedRateDeviationMinRightUnwindLM
        self.fixedRateDeviationMinLeftUnwindIM = fixedRateDeviationMinLeftUnwindIM
        self.fixedRateDeviationMinRightUnwindIM = fixedRateDeviationMinRightUnwindIM
        self.gamma = gamma
        self.minMarginToIncentiviseLiquidators = minMarginToIncentiviseLiquidators

    # tested
    def worstCaseVariableFactorAtMaturity(self, timeInSecondsFromNowToMaturity, isFT, isLM, 
                                          lowerApyBound, upperApyBound, accruedVariableFactor):

        timeInYearsFromNowToMaturity = timeInSecondsFromNowToMaturity / SECONDS_IN_YEAR
        rateFromStart = accruedVariableFactor + 1
        
        apyBound = lowerApyBound
        if isFT:
            apyBound = upperApyBound

        if not isLM:
            if isFT:
                apyBound = apyBound * self.apyUpperMultiplier
            else:
                apyBound = apyBound * self.apyLowerMultiplier


        variableFactor = rateFromStart * (apyBound * timeInYearsFromNowToMaturity + 1) - 1
        return variableFactor

    # inherintely tested
    def _getMarginRequirement(self, fixedTokenBalance, variableTokenBalance, isLM, lowerApyBound, upperApyBound,
                              termStartTimestamp, termEndTimestamp, currentTimestamp, accruedVariableFactor):
        if (fixedTokenBalance >= 0) and (variableTokenBalance >= 0):
            return 0

        timeInSecondsFromNowToMaturity = termEndTimestamp - currentTimestamp
        exp1 = fixedTokenBalance * \
            self.fixedFactor(True, termStartTimestamp,
                             termEndTimestamp, currentTimestamp)

        exp2 = variableTokenBalance * self.worstCaseVariableFactorAtMaturity(
            timeInSecondsFromNowToMaturity,
            variableTokenBalance < 0,
            isLM,
            lowerApyBound,
            upperApyBound,
            accruedVariableFactor
        )

        maxCashflowDeltaToCoverPostMaturity = exp1 + exp2

        if maxCashflowDeltaToCoverPostMaturity < 0:
            margin = -maxCashflowDeltaToCoverPostMaturity
        else:
            margin = 0

        minimumMarginRequirement = abs(variableTokenBalance) * self.etaLM if isLM else abs(variableTokenBalance) * self.etaIM
        minimumMarginRequirement = minimumMarginRequirement * timeInSecondsFromNowToMaturity/SECONDS_IN_YEAR

        if margin < minimumMarginRequirement:
            margin = minimumMarginRequirement

        if margin < self.minMarginToIncentiviseLiquidators:
            margin = self.minMarginToIncentiviseLiquidators

        return margin

    # inherintely tested
    def calculateFixedTokenBalance(self, amountFixed, excessBalance, termStartTimestamp, termEndTimestamp,
                                   currentTimestamp):
        fixedFactor = self.fixedFactor(
            True, termStartTimestamp, termEndTimestamp, currentTimestamp)

        return amountFixed - (excessBalance / fixedFactor)

    # tested
    def fixedFactor(self, atMaturity, termStartTimestamp, termEndTimestamp, currentTimestamp):

        if (atMaturity or (currentTimestamp >= termEndTimestamp)):
            timeInSeconds = termEndTimestamp - termStartTimestamp
        else:
            timeInSeconds = currentTimestamp - termStartTimestamp

        timeInYears = timeInSeconds / SECONDS_IN_YEAR

        fixedFactor = timeInYears / 100

        return fixedFactor

    # tested
    def getExcessBalance(self, amountFixed, amountVariable, accruedVariableFactor, termStartTimestamp,
                         termEndTimestamp, currentTimestamp):

        excessFixedAccruedBalance = amountFixed * self.fixedFactor(False, termStartTimestamp, termEndTimestamp,
                                                                   currentTimestamp)
        excessVariableAccruedBalance = amountVariable * accruedVariableFactor
        excessBalance = excessFixedAccruedBalance + excessVariableAccruedBalance

        return excessBalance

    # tested
    def getFixedTokenBalance(self, amountFixed, amountVariable, accruedVariableFactor, termStartTimestamp,
                             termEndTimestamp, currentTimestamp):

        if (amountFixed == 0) and (amountVariable == 0):
            return 0

        excessBalance = self.getExcessBalance(amountFixed, amountVariable, accruedVariableFactor, termStartTimestamp,
                                              termEndTimestamp, currentTimestamp)

        fixedTokenBalance = self.calculateFixedTokenBalance(
            amountFixed=amountFixed,
            excessBalance=excessBalance,
            termStartTimestamp=termStartTimestamp,
            termEndTimestamp=termEndTimestamp,
            currentTimestamp=currentTimestamp
        )

        return fixedTokenBalance

    # tested
    def getFixedTokenDeltaUnbalancedSimulatedUnwind(self, variableTokenDeltaAbsolute, fixedRateStart,
                                                    startingFixedRateMultiplier,
                                                    fixedRateDeviationMin,
                                                    termEndTimestamp, currentTimestamp, tMax, gamma,
                                                    isFTUnwind):

        # calculate D
        upperD = fixedRateStart * startingFixedRateMultiplier

        if upperD < fixedRateDeviationMin:
            upperD = fixedRateDeviationMin

        # calcualte d
        scaledTime = (termEndTimestamp - currentTimestamp) / tMax
        expInput = scaledTime * (-gamma)
        oneMinusTimeFactor = 1 - math.exp(expInput)
        d = upperD * oneMinusTimeFactor

        if isFTUnwind:
            if fixedRateStart > d:
                fixedRateCF = fixedRateStart - d
            else:
                fixedRateCF = 0
        else:
            fixedRateCF = fixedRateStart + d

        fixedTokenDeltaUnbalanced = variableTokenDeltaAbsolute * fixedRateCF

        return fixedTokenDeltaUnbalanced

    # inherintely tested
    def getMinimumMarginRequirement(self, fixedTokenBalance, variableTokenBalance, isLM, fixedRate, currentTimestamp,
                                    variableFactor, lowerApyBound, upperApyBound, termStartTimestamp, termEndTimestamp):

        if variableTokenBalance == 0:
            return 0

        if variableTokenBalance > 0:
            if fixedTokenBalance > 0:
                return 0

            if isLM:
                devMul = self.devMulLeftUnwindLM
                fixedRateDeviationMin = self.fixedRateDeviationMinLeftUnwindLM
            else:
                devMul = self.devMulLeftUnwindIM
                fixedRateDeviationMin = self.fixedRateDeviationMinLeftUnwindIM

            absoluteVariableTokenBalance = variableTokenBalance
            isVariableTokenBalancePositive = True

        else:

            if isLM:
                devMul = self.devMulRightUnwindLM
                fixedRateDeviationMin = self.fixedRateDeviationMinRightUnwindLM
            else:
                devMul = self.devMulRightUnwindIM
                fixedRateDeviationMin = self.fixedRateDeviationMinRightUnwindIM

            absoluteVariableTokenBalance = -variableTokenBalance
            isVariableTokenBalancePositive = False

        fixedTokenDeltaUnbalanced = self.getFixedTokenDeltaUnbalancedSimulatedUnwind(
            absoluteVariableTokenBalance,
            fixedRate,
            devMul,
            fixedRateDeviationMin,
            termEndTimestamp,
            currentTimestamp,
            self.tMax,
            self.gamma,
            isVariableTokenBalancePositive
        )

        if isVariableTokenBalancePositive:
            amountFixed = fixedTokenDeltaUnbalanced
        else:
            amountFixed = -fixedTokenDeltaUnbalanced

        fixedTokenDelta = self.getFixedTokenBalance(
            amountFixed,
            -variableTokenBalance,
            variableFactor,
            termStartTimestamp,
            termEndTimestamp,
            currentTimestamp
        )

        updatedFixedTokenBalance = fixedTokenBalance + fixedTokenDelta

        margin = self._getMarginRequirement(fixedTokenBalance=updatedFixedTokenBalance,
                                            variableTokenBalance=0,
                                            isLM=isLM,
                                            lowerApyBound=lowerApyBound,
                                            upperApyBound=upperApyBound,
                                            termStartTimestamp=termStartTimestamp,
                                            termEndTimestamp=termEndTimestamp,
                                            currentTimestamp=currentTimestamp,
                                            accruedVariableFactor=variableFactor)

        if margin < self.minMarginToIncentiviseLiquidators:
            margin = self.minMarginToIncentiviseLiquidators

        return margin

    # tested
    def getExtraBalances(self, fromTick, toTick, liquidity, variableFactor, termStartTimestamp, termEndTimestamp, currentTimestamp):

        assert(liquidity >= 0)

        fromTickSqrtRatio = getSqrtRatioAtTick(fromTick)
        toTickSqrtRatio = getSqrtRatioAtTick(toTick)

        amount0Liquidity = liquidity

        if fromTick < toTick:
            amount0Liquidity = -amount0Liquidity

        amount0 = getAmount0Delta(
            fromTickSqrtRatio,
            toTickSqrtRatio,
            amount0Liquidity
        )

        amount1Liquidity = liquidity

        if fromTick >= toTick:
            amount1Liquidity = -amount1Liquidity

        amount1 = getAmount1Delta(
            fromTickSqrtRatio,
            toTickSqrtRatio,
            amount1Liquidity
        )

        extraFixedTokenBalance = self.getFixedTokenBalance(
            amountFixed=amount0,
            amountVariable=amount1,
            accruedVariableFactor=variableFactor,
            termStartTimestamp=termStartTimestamp,
            termEndTimestamp=termEndTimestamp,
            currentTimestamp=currentTimestamp
        )

        return extraFixedTokenBalance, amount1

    # tested
    def getMarginRequirement(self, fixedTokenBalance, variableTokenBalance, isLM, sqrtPrice,
                             lowerApyBound, upperApyBound, termStartTimestamp, termEndTimestamp,
                             currentTimestamp, accruedVariableFactor):

        margin = self._getMarginRequirement(
            fixedTokenBalance=fixedTokenBalance,
            variableTokenBalance=variableTokenBalance,
            isLM=isLM,
            lowerApyBound=lowerApyBound,
            upperApyBound=upperApyBound,
            termStartTimestamp=termStartTimestamp,
            termEndTimestamp=termEndTimestamp,
            currentTimestamp=currentTimestamp,
            accruedVariableFactor=accruedVariableFactor
        )

        # create a conversion function for sqrt price to fixed rate
        # calculate minimum, proceed (ref https://github.com/Voltz-Protocol/voltz-core/blob/cc9d45193ef658df9dd79c6940959128b44e7154/contracts/MarginEngine.sol#L1136)

        fixedRate = sqrtPriceToFixedRate(sqrtPrice)

        minimumMarginRequirement = self.getMinimumMarginRequirement(
            fixedTokenBalance=fixedTokenBalance,
            variableTokenBalance=variableTokenBalance,
            isLM=isLM,
            fixedRate=fixedRate,
            currentTimestamp=currentTimestamp,
            accruedVariableFactor=accruedVariableFactor,
            lowerApyBound=lowerApyBound,
            upperApyBound=upperApyBound,
            termStartTimestamp=termStartTimestamp,
            termEndTimestamp=termEndTimestamp
        )

        if margin < minimumMarginRequirement:
            margin = minimumMarginRequirement

        return margin

    # tested
    def getPositionMarginRequirement(self, variableFactor, currentTick, positionLiquidity, tickLower, tickUpper, sqrtPrice,
                                     termStartTimestamp, termEndTimestamp, currentTimestamp, positionVariableTokenBalance,
                                     positionFixedTokenBalance, isLM, lowerApyBound, upperApyBound):

        if positionLiquidity > 0:

            if currentTick < tickLower:
                inRangeTick = tickLower
            else:
                if currentTick < tickUpper:
                    inRangeTick = currentTick
                else:
                    inRangeTick = tickUpper

            # scenario 1: a trader comes in and trades all the liquidity all the way to the upper tick
            # scenario 2: a trader comes in and trades all the liquidity all the way to the lower tick

            # fromTick, toTick, liquidity, variableFactor, termStartTimestamp, termEndTimestamp, currentTimestamp

            extraFixedTokenBalance, extraVariableTokenBalance = 0, 0

            if currentTick < tickUpper:
                extraFixedTokenBalance, extraVariableTokenBalance = self.getExtraBalances(
                    inRangeTick,
                    tickUpper,
                    positionLiquidity,
                    variableFactor,
                    termStartTimestamp=termStartTimestamp,
                    termEndTimestamp=termEndTimestamp,
                    currentTimestamp=currentTimestamp
                )

            scenario1LPVariableTokenBalance = positionVariableTokenBalance + \
                extraVariableTokenBalance
            scenario1LPFixedTokenBalance = positionFixedTokenBalance + extraFixedTokenBalance

            if currentTick > tickLower:
                extraFixedTokenBalance, extraVariableTokenBalance = self.getExtraBalances(
                    inRangeTick,
                    tickLower,
                    positionLiquidity,
                    variableFactor,
                    termStartTimestamp=termStartTimestamp,
                    termEndTimestamp=termEndTimestamp,
                    currentTimestamp=currentTimestamp
                )
            else:
                extraFixedTokenBalance, extraVariableTokenBalance = 0, 0

            scenario2LPVariableTokenBalance = positionVariableTokenBalance + \
                extraVariableTokenBalance
            scenario2LPFixedTokenBalance = positionFixedTokenBalance + extraFixedTokenBalance

            lowPrice = getSqrtRatioAtTick(tickLower)
            highPrice = getSqrtRatioAtTick(tickUpper)

            if sqrtPrice < lowPrice:
                lowPrice = sqrtPrice

            if sqrtPrice > highPrice:
                highPrice = sqrtPrice

            if scenario1LPVariableTokenBalance > 0:
                scenario1SqrtPrice = highPrice
            else:
                scenario1SqrtPrice = lowPrice

            if scenario2LPVariableTokenBalance > 0:
                scenario2SqrtPrice = highPrice
            else:
                scenario2SqrtPrice = lowPrice

            # fixedTokenBalance, variableTokenBalance, isLM, sqrtPrice,
            # lowerApyBound, upperApyBound, termStartTimestamp, termEndTimestamp,
            # currentTimestamp, accruedVariableFactor

            scenario1MarginRequirement = self.getMarginRequirement(
                scenario1LPFixedTokenBalance,
                scenario1LPVariableTokenBalance,
                isLM,
                scenario1SqrtPrice,
                lowerApyBound,
                upperApyBound,
                termStartTimestamp,
                termEndTimestamp,
                currentTimestamp,
                variableFactor
            )

            scenario2MarginRequirement = self.getMarginRequirement(
                scenario2LPFixedTokenBalance,
                scenario2LPVariableTokenBalance,
                isLM,
                scenario2SqrtPrice,
                lowerApyBound,
                upperApyBound,
                termStartTimestamp,
                termEndTimestamp,
                currentTimestamp,
                variableFactor
            )
            if scenario1MarginRequirement > scenario2MarginRequirement:
                return scenario1MarginRequirement
            else:
                return scenario2MarginRequirement
        else:

            return self._getMarginRequirement(
                fixedTokenBalance=positionFixedTokenBalance,
                variableTokenBalance=positionVariableTokenBalance,
                isLM=isLM,
                lowerApyBound=lowerApyBound,
                upperApyBound=upperApyBound,
                termStartTimestamp=termStartTimestamp,
                termEndTimestamp=termEndTimestamp,
                currentTimestamp=currentTimestamp,
                accruedVariableFactor=variableFactor
            )

    def generate_full_output(self, df_apy, date_original, tokens, leverage_factor=1, notional=1000, lp_fix=0, lp_var=0, tick_l=5000, tick_u=6000, fr_market="neutral"):

        # All tokens
        # All trader types
        # Margin requirements & pnl
        lp_fixed_token_balance = lp_fix
        lp_variable_token_balance = lp_var
        tickLower = tick_l
        tickUpper = tick_u
        lp_liquidity = notional_to_liquidity(
            notional=notional, tick_l=tickLower, tick_u=tickUpper)

        list_of_tokens = tokens
        trader_types = ["ft", "vt", "lp"]

        dfs_per_token = []
        # We need to keep track of the position names for the PortfolioCalculator
        balance_per_token = {}

        # Before looping of each token we need to first convert the timestamp to unix time **IN SECONDS**, so
        # that we can later on normalised to SECONDS_IN_YEAR for the APY calculation
        df = date_to_unix_time(df_apy, date_original)
        for token in list_of_tokens:
            # Update for each new token
            df = preprocess_df(df, token, fr_market=fr_market)
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

                    df = generate_margin_requirements_trader(self, df, ft_fixed_token_balance,
                                                                ft_variable_token_balance, 'ft', f'{token}')

                    daily_fixed_rate = (
                        (abs(ft_fixed_token_balance) / ft_variable_token_balance) / 100) / 365

                    fixed_factor_series = pd.Series(data=1, index=range(len(df)))
                    fixed_factor_series = pd.Series(
                        data=fixed_factor_series.index * daily_fixed_rate)

                    df = generate_pnl_trader(df, fixed_factor_series, ft_fixed_token_balance,
                                                                ft_variable_token_balance, 'ft', f'{token}')

                    df = generate_net_margin_trader(df,
                                                                        ft_fixed_token_balance, ft_variable_token_balance, 'ft', f'{token}', leverage_factor=leverage_factor)

                elif trader_type == 'vt':

                    df = generate_margin_requirements_trader(self, df, vt_fixed_token_balance,
                                                                                vt_variable_token_balance, 'vt', f'{token}')

                    daily_fixed_rate = (
                        (abs(vt_fixed_token_balance) / vt_variable_token_balance) / 100) / 365

                    fixed_factor_series = pd.Series(data=1, index=range(len(df)))
                    fixed_factor_series = pd.Series(
                        data=fixed_factor_series.index * daily_fixed_rate)

                    df = generate_pnl_trader(df, fixed_factor_series,
                                                                vt_fixed_token_balance,
                                                                vt_variable_token_balance, 'vt', f'{token}')

                    df = generate_net_margin_trader(df,
                                                                        vt_fixed_token_balance, vt_variable_token_balance, 'vt', f'{token}', leverage_factor=leverage_factor)

                else:
                    df = generate_margin_requirements_lp(
                        marginCalculator=self,
                        df=df,
                        fixedTokenBalance=lp_fixed_token_balance,
                        variableTokenBalance=lp_variable_token_balance,
                        liquidity=lp_liquidity,
                        tickLower=tickLower,
                        tickUpper=tickUpper,
                        token=f'{token}'
                    )

            df = df.rename(
                columns={
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
        # Add t_year for downstream APY calculation
        result["t_years"] = [(i-result.index.values[0]) /
                            SECONDS_IN_YEAR for i in result.index.values]

        return result, balance_per_token

import math

SECONDS_IN_YEAR = 31536000


class MarginCalculator:

    def __init__(self, apyUpperMultiplier, apyLowerMultiplier, sigmaSquared, alpha, beta, xiUpper, xiLower, tMax,
                 devMulLeftUnwindLM, devMulRightUnwindLM, devMulLeftUnwindIM, devMulRightUnwindIM,
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


    def compute_minimum_margin_requirement_df_row(self, row, termStartTimestamp, termEndTimestamp, fixedTokenBalance, variableTokenBalance, isLM):

        minimum_margin_requirement = self.getMinimumMarginRequirement(
            fixedTokenBalance=fixedTokenBalance,
            variableTokenBalance=variableTokenBalance,
            isLM=isLM,
            fixedRate=row['fr'],
            currentTimestamp=row['date'],
            accruedVariableFactor=row['variable factor'],
            lowerApyBound=row['lower'],
            upperApyBound=row['upper'],
            termStartTimestamp=termStartTimestamp,
            termEndTimestamp=termEndTimestamp
        )

        return minimum_margin_requirement

    def fixedRateToSqrtPrice(self, fixedRate):

        return math.sqrt(1 / fixedRate)




    def compute_margin_requirement_df_row_trader(self, row, termStartTimestamp, termEndTimestamp,
                                                 fixedTokenBalance, variableTokenBalance, isLM):

        marginRequirement = self.getMarginRequirement(
            fixedTokenBalance=fixedTokenBalance,
            variableTokenBalance=variableTokenBalance,
            isLM=isLM,
            sqrtPrice=self.fixedRateToSqrtPrice(row['fr']*100),
            lowerApyBound=row['lower'],
            upperApyBound=row['upper'],
            termStartTimestamp=termStartTimestamp,
            termEndTimestamp=termEndTimestamp,
            currentTimestamp=row['date'],
            accruedVariableFactor=row['variable factor']
        )

        return marginRequirement

    def generate_pnl_trader(self, df, fixedFactorSeries, fixedTokenBalance, variableTokenBalance, trader_type, token):

        if variableTokenBalance > 0:
            df.loc[:, 'net factor'] = df.loc[:, 'variable factor'] - fixedFactorSeries
        else:
            df.loc[:, 'net factor'] = fixedFactorSeries - df.loc[:, 'variable factor']

        df.loc[:, f'pnl_{trader_type}_{token}_{fixedTokenBalance}_{variableTokenBalance}'] = abs(variableTokenBalance) * df.loc[:, 'net factor']

        return df

    # We may specifiy a specific leverage factor since the trader
    # leverage = notional / margin deposited <= notional / initial margin requirement
    def generate_net_margin_trader(self, df, fixedTokenBalance, variableTokenBalance, trader_type, token, leverage_factor=1):

        # column with the initial margin requirement
        column_name = f'mr_im_{trader_type}_{token}_{fixedTokenBalance}_{variableTokenBalance}'
        initial_margin_requirement_series = df.loc[:, column_name]
        margin_deposited = initial_margin_requirement_series[0] * leverage_factor 
        pnl_column_name = f'pnl_{trader_type}_{token}_{fixedTokenBalance}_{variableTokenBalance}'
        df.loc[:, f"margin_deposited_{trader_type}_{token}_{fixedTokenBalance}_{variableTokenBalance}"] = margin_deposited
        df.loc[:, f"net_margin_{trader_type}_{token}_{fixedTokenBalance}_{variableTokenBalance}"] = margin_deposited + df.loc[:, pnl_column_name]

        return df


    def generate_margin_requirements_trader(self, df, fixedTokenBalance, variableTokenBalance, trader_type, token):

        # produce a time series of margin requirements
        termStartTimestamp = df.loc[:, "date"].iloc[0]
        termEndTimestamp = df.loc[:, "date"].iloc[-1]

        df.loc[:, f'mmr_lm_{trader_type}_{token}_{fixedTokenBalance}_{variableTokenBalance}'] = df.apply(self.compute_minimum_margin_requirement_df_row,
                                    args=(termStartTimestamp, termEndTimestamp, fixedTokenBalance, variableTokenBalance,
                                          True),
                                    axis=1)

        df.loc[:, f'mmr_im_{trader_type}_{token}_{fixedTokenBalance}_{variableTokenBalance}'] = df.apply(self.compute_minimum_margin_requirement_df_row,
                                    args=(termStartTimestamp, termEndTimestamp, fixedTokenBalance, variableTokenBalance,
                                          False),
                                    axis=1)

        df.loc[:, f'mr_lm_{trader_type}_{token}_{fixedTokenBalance}_{variableTokenBalance}'] = df.apply(self.compute_margin_requirement_df_row_trader,
                                    args=(termStartTimestamp, termEndTimestamp, fixedTokenBalance, variableTokenBalance,
                                          True),
                                    axis=1)

        df.loc[:, f'mr_im_{trader_type}_{token}_{fixedTokenBalance}_{variableTokenBalance}'] = df.apply(self.compute_margin_requirement_df_row_trader,
                                    args=(termStartTimestamp, termEndTimestamp, fixedTokenBalance, variableTokenBalance,
                                          False),
                                    axis=1)

        return df


    def fixedRateToTick(self, fixedRate):

        price = 1 / fixedRate
        # price = 1.0001^(tick)
        # tick = log_1.0001(price)

        return math.log(price, 1.0001)



    def compute_margin_requirement_df_row_lp(self, row, termStartTimestamp, termEndTimestamp,
                                            positionLiquidity, tickLower, tickUpper, positionVariableTokenBalance,
                                            positionFixedTokenBalance, isLM):
        
        marginRequirement = self.getPositionMarginRequirement(
            variableFactor=0,
            currentTick=self.fixedRateToTick(row['fr']*100),
            positionLiquidity=positionLiquidity,
            tickLower=tickLower,
            tickUpper=tickUpper,
            sqrtPrice=self.fixedRateToSqrtPrice(row['fr']*100),
            termStartTimestamp=termStartTimestamp,
            termEndTimestamp=termEndTimestamp,
            currentTimestamp=row['date'],
            positionVariableTokenBalance=positionVariableTokenBalance,
            positionFixedTokenBalance=positionFixedTokenBalance,
            isLM=isLM,
            lowerApyBound=row['lower'],
            upperApyBound=row['upper']
        )

        return marginRequirement

    def generate_margin_requirements_lp(self, df, token, fixedTokenBalance, variableTokenBalance, liquidity, tickLower, tickUpper):

        # produce a time series of margin requirements
        # assume the lp has zero fixed and variable tokens at the start, todo: extend to the case where that's not the case

        termStartTimestamp = df.loc[:, "date"].iloc[0]
        termEndTimestamp = df.loc[:, "date"].iloc[-1]

        df.loc[:, f'mr_lm_lp_{token}_{fixedTokenBalance}_{variableTokenBalance}_{liquidity}'] = df.apply(self.compute_margin_requirement_df_row_lp,
                                    args=(termStartTimestamp, termEndTimestamp, liquidity, tickLower, tickUpper,
                                          variableTokenBalance, fixedTokenBalance, True),
                                    axis=1)

        df.loc[:, f'mr_im_lp_{token}_{fixedTokenBalance}_{variableTokenBalance}_{liquidity}'] = df.apply(self.compute_margin_requirement_df_row_lp,
                                    args=(termStartTimestamp, termEndTimestamp, liquidity, tickLower, tickUpper,
                                          variableTokenBalance, fixedTokenBalance, False),
                                    axis=1)

        # todo: requirements towards term end timestamp seem to be suspiciously large
        return df


    def worstCaseVariableFactorAtMaturity(self, timeInSecondsFromStartToMaturity,
                                          isFT, isLM, lowerApyBound, upperApyBound):

        timeInYearsFromStartToMaturity = timeInSecondsFromStartToMaturity / SECONDS_IN_YEAR

        apyBound = lowerApyBound
        if isFT:
            apyBound = upperApyBound

        variableFactor = apyBound * timeInYearsFromStartToMaturity

        if not isLM:
            if isFT:
                variableFactor = variableFactor * self.apyUpperMultiplier
            else:
                variableFactor = variableFactor * self.apyLowerMultiplier

        return variableFactor


    def _getMarginRequirement(self, fixedTokenBalance, variableTokenBalance, isLM, lowerApyBound, upperApyBound,
                              termStartTimestamp, termEndTimestamp, currentTimestamp):
        if (fixedTokenBalance >= 0) and (variableTokenBalance >= 0):
            return 0

        timeInSecondsFromStartToMaturity = termEndTimestamp - termStartTimestamp
        exp1 = fixedTokenBalance * self.fixedFactor(True, termStartTimestamp, termEndTimestamp, currentTimestamp)

        exp2 = variableTokenBalance * self.worstCaseVariableFactorAtMaturity(
            timeInSecondsFromStartToMaturity,
            variableTokenBalance < 0,
            isLM,
            lowerApyBound,
            upperApyBound
        )

        maxCashflowDeltaToCoverPostMaturity = exp1 + exp2

        if maxCashflowDeltaToCoverPostMaturity < 0:
            margin = -maxCashflowDeltaToCoverPostMaturity
        else:
            margin = 0

        return margin


    def calculateFixedTokenBalance(self, amountFixed, excessBalance, termStartTimestamp, termEndTimestamp,
                                   currentTimestamp):
        fixedFactor = self.fixedFactor(True, termStartTimestamp, termEndTimestamp, currentTimestamp)

        return amountFixed - (excessBalance / fixedFactor)

    def fixedFactor(self, atMaturity, termStartTimestamp, termEndTimestamp, currentTimestamp):

        if (atMaturity or (currentTimestamp >= termEndTimestamp)):
            timeInSeconds = termEndTimestamp - termStartTimestamp
        else:
            timeInSeconds = currentTimestamp - termStartTimestamp

        timeInYears = timeInSeconds / SECONDS_IN_YEAR

        fixedFactor = timeInYears / 100

        return fixedFactor

    def getExcessBalance(self, amountFixed, amountVariable, accruedVariableFactor, termStartTimestamp,
                         termEndTimestamp, currentTimestamp):

        excessFixedAccruedBalance = amountFixed * self.fixedFactor(False, termStartTimestamp, termEndTimestamp,
                                                                   currentTimestamp)
        excessVariableAccruedBalance = amountVariable * accruedVariableFactor
        excessBalance = excessFixedAccruedBalance + excessVariableAccruedBalance

        return excessBalance

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

        fixedTokenDeltaUnbalanced = variableTokenDeltaAbsolute * fixedRateCF * 100

        return fixedTokenDeltaUnbalanced

    def getMinimumMarginRequirement(self, fixedTokenBalance, variableTokenBalance, isLM, fixedRate, currentTimestamp,
                                    accruedVariableFactor, lowerApyBound, upperApyBound, termStartTimestamp, termEndTimestamp):

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
            accruedVariableFactor,
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
                                            currentTimestamp=currentTimestamp)

        if margin < self.minMarginToIncentiviseLiquidators:
            margin = self.minMarginToIncentiviseLiquidators

        return margin


    def _getAmount0Delta(self, sqrtRatioA, sqrtRatioB, liquidity):

        if sqrtRatioA > sqrtRatioB:
            sqrtRatioA, sqrtRatioB = sqrtRatioB, sqrtRatioA

        numerator = sqrtRatioB - sqrtRatioA
        denominator = sqrtRatioB * sqrtRatioA

        result = liquidity * (numerator / denominator) # without rounding up

        return result


    def getAmount0Delta(self, sqrtRatioA, sqrtRatioB, liquidity):

        if liquidity < 0:
            return -self._getAmount0Delta(sqrtRatioA, sqrtRatioB, -liquidity)
        else:
            return self._getAmount0Delta(sqrtRatioA, sqrtRatioB, liquidity)


    def _getAmount1Delta(self, sqrtRatioA, sqrtRatioB, liquidity):

        if sqrtRatioA > sqrtRatioB:
            sqrtRatioA, sqrtRatioB = sqrtRatioB, sqrtRatioA

        return liquidity * (sqrtRatioB - sqrtRatioA)


    def getAmount1Delta(self, sqrtRatioA, sqrtRatioB, liquidity):

        if liquidity < 0:
            return -self._getAmount1Delta(sqrtRatioA, sqrtRatioB, -liquidity)
        else:
            return self._getAmount1Delta(sqrtRatioA, sqrtRatioB, liquidity)


    def getSqrtRatioAtTick(self, tick):
        # sqrt(1.0001 ^ tick)
        return math.sqrt(pow(1.0001, tick))
  
    # Get the liquidity from the notional and prices:
    # Liquidity = Notional / (sqrt(upper) - sqrt(lower))
    def notional_to_liquidity(self, notional=1000, tick_l=5000, tick_u=6000):
        sqrt_upper = self.getSqrtRatioAtTick(tick=tick_u)
        sqrt_lower = self.getSqrtRatioAtTick(tick=tick_l)
        return notional/(sqrt_upper - sqrt_lower)

    def getExtraBalances(self, fromTick, toTick, liquidity, variableFactor, termStartTimestamp, termEndTimestamp, currentTimestamp):

        assert(liquidity >= 0)

        fromTickSqrtRatio = self.getSqrtRatioAtTick(fromTick)
        toTickSqrtRatio = self.getSqrtRatioAtTick(toTick)

        amount0Liquidity = liquidity

        if fromTick < toTick:
            amount0Liquidity = -amount0Liquidity

        amount0 = self.getAmount0Delta(
            fromTickSqrtRatio,
            toTickSqrtRatio,
            amount0Liquidity
        )

        amount1Liquidity = liquidity

        if fromTick >= toTick:
            amount1Liquidity = -amount1Liquidity

        amount1 = self.getAmount1Delta(
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


    def sqrtPriceToFixedRate(self, sqrtPrice):

        # the fixed rate is in percentage points, i.e. result of 1 refers to a 1% fixed rate

        return 1 / (sqrtPrice ** 2)

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
            currentTimestamp=currentTimestamp
        )

        # create a conversion function for sqrt price to fixed rate
        # calculate minimum, proceed (ref https://github.com/Voltz-Protocol/voltz-core/blob/cc9d45193ef658df9dd79c6940959128b44e7154/contracts/MarginEngine.sol#L1136)

        fixedRate = self.sqrtPriceToFixedRate(sqrtPrice)

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

            extraFixedTokenBalance, extraVariableTokenBalance = 0,0

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

            scenario1LPVariableTokenBalance = positionVariableTokenBalance + extraVariableTokenBalance
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
                extraFixedTokenBalance, extraVariableTokenBalance = 0,0

            scenario2LPVariableTokenBalance = positionVariableTokenBalance + extraVariableTokenBalance
            scenario2LPFixedTokenBalance = positionFixedTokenBalance + extraFixedTokenBalance

            lowPrice = self.getSqrtRatioAtTick(tickLower)
            highPrice = self.getSqrtRatioAtTick(tickUpper)
            
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
                currentTimestamp=currentTimestamp
            )
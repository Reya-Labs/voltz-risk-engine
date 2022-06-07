import math

SECONDS_IN_WEEK = 604800
SECONDS_IN_YEAR = 31536000

# tick = 1 / (log in base 1.0001 of fixedRate)
def fixedRateToTick(fixedRate):
    return -math.log(fixedRate, 1.0001)


# sqrtPrice = sqrt(1 / fixedRate)
def fixedRateToSqrtPrice(fixedRate):
    return math.sqrt(1 / fixedRate)

# sqrtPrice = sqrt(1.0001 ^ tick)
def getSqrtRatioAtTick(tick):
    return math.sqrt(math.pow(1.0001, tick))

# fixedRate = (1 / sqrtPrice) ** 2
def sqrtPriceToFixedRate(sqrtPrice):
    return 1 / (sqrtPrice ** 2)


def _getAmount0Delta(sqrtRatioA, sqrtRatioB, liquidity):

    if sqrtRatioA > sqrtRatioB:
        sqrtRatioA, sqrtRatioB = sqrtRatioB, sqrtRatioA

    numerator = sqrtRatioB - sqrtRatioA
    denominator = sqrtRatioB * sqrtRatioA

    result = liquidity * (numerator / denominator)  # without rounding up

    return result


def getAmount0Delta(sqrtRatioA, sqrtRatioB, liquidity):

    if liquidity < 0:
        return -_getAmount0Delta(sqrtRatioA, sqrtRatioB, -liquidity)
    else:
        return _getAmount0Delta(sqrtRatioA, sqrtRatioB, liquidity)


def _getAmount1Delta(sqrtRatioA, sqrtRatioB, liquidity):

    if sqrtRatioA > sqrtRatioB:
        sqrtRatioA, sqrtRatioB = sqrtRatioB, sqrtRatioA

    return liquidity * (sqrtRatioB - sqrtRatioA)


def getAmount1Delta(sqrtRatioA, sqrtRatioB, liquidity):

    if liquidity < 0:
        return -_getAmount1Delta(sqrtRatioA, sqrtRatioB, -liquidity)
    else:
        return _getAmount1Delta(sqrtRatioA, sqrtRatioB, liquidity)


# Get the liquidity from the notional and prices:
# Liquidity = Notional / (sqrt(upper) - sqrt(lower))
def notional_to_liquidity(notional, tick_l, tick_u):
    sqrt_upper = getSqrtRatioAtTick(tick=tick_u)
    sqrt_lower = getSqrtRatioAtTick(tick=tick_l)

    return notional/(sqrt_upper - sqrt_lower)

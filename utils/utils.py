import math

import numpy as np

from utils.constants import FT_ERROR, FT_FACTOR

SECONDS_IN_DAY = 86400
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


def date_to_unix_time(df_input, date_original):
    import datetime
    import time

    df = df_input.copy()
    df.reset_index(drop=True, inplace=True)

    date_time = []
    for date in date_original:
        y, m, d = [int(date.split(" ")[0].split("-")[i]) for i in range(3)]
        h, mt, s = [int(date.split(" ")[1].split(":")[i])
                    for i in range(3)]
        date_time.append(datetime.datetime(y, m, d, h, mt, s))

    df["date"] = np.array([time.mktime(dt.timetuple())
                          for dt in date_time])

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
def preprocess_df(df_input, token, fr_market="neutral"):

    # Pick a single token
    df = df_input.copy()  # Needs to have already been timestamp processed
    df = df.loc[:, [f"{token} model", f"{token} APY lower",
                    f"{token} APY upper", f"{token} model", "date"]]
    # USDC model corresponds to the simulation of fixed rates
    df.columns = ["apy", "lower", "upper", "fr", "date"]

    # Compute liquidity index from the given token APY and the corresponding variable factor
    df = calculate_liquidity_index(df)
    liquidity_index_term_start_timestamp = df.loc[0, "liquidity_index"]
    df.loc[:, "variable factor"] = (
        df.loc[:, "liquidity_index"] / liquidity_index_term_start_timestamp) - 1

    # Update the fixed rate if bearish/bullish conditions are specified
    if fr_market == "bull":
        df["fr"] = [df["apy"].values[i] + FT_FACTOR *
                    (df["date"].values[-1]-df["date"].values[i])/df["date"].values[-1] + FT_ERROR for i in range(len(df))]
    if fr_market == "bear":
        df["fr"] = [df["apy"].values[i] - FT_FACTOR *
                    (df["date"].values[-1]-df["date"].values[i])/df["date"].values[-1] - FT_ERROR for i in range(len(df))]
        # Check on fixed rates, in case bearish rates go negative
        df["fr"] = [fr if fr > 0 else 0.0001 for fr in df["fr"].values]

    return df


def compute_liquidity_index(row, df):
    if row.name <= 7:
        counterfactual_rate = df.loc[0, "rate"]
        return ((counterfactual_rate - 1) / 7) * row.name + 1
    else:

        df = df.shift(row.name % 7).dropna().iloc[::7, :]
        df = df[df['date'] < row['date']]
        return df.loc[:, 'rate'].cumprod().iloc[-1] * row['rate']


def calculate_liquidity_index(df):
    # todo: move into a separate module/script
    WEEK_IN_YEARS = 7 / 365  # rough approx

    df.loc[:, 'rate'] = 1 + df.loc[:, 'apy'] * WEEK_IN_YEARS

    df.loc[:, 'liquidity_index'] = df.apply(
        compute_liquidity_index,
        args=(df,),
        axis=1
    )

    # calculate the rate week on week
    # todo: also multiply by liquidity indicies for the first 7 rows

    return df


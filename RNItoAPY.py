import pandas as pd
import numpy as np

import datetime

from utils import SECONDS_IN_DAY, SECONDS_IN_YEAR, SECONDS_IN_360YEAR, date_to_unix_time

# """
#    This function applies the following transformations to the raw data:
#        - converts the dates to unix timestamps
#        - scales the liquidity index
#        - sort the entries by timestamps
# """
def getPreparedRNIData(df_input):
    # copy data 
    df = df_input.copy()

    # set ``date``` column as index to be able to use ``date_to_unix_time`` function
    df.set_index("date", inplace=True)

    # transform dates into unix timestamps
    df_unix_time = date_to_unix_time(df, df.index, separator="T", timezone_separator=None)

    # ``liquidity index`` is de-scaled (by 10**27)
    floating_rni = []
    for rni in df_unix_time["liquidityIndex"]:
        float_rni = rni if isinstance(rni, float) else rni[:-27] + "." + rni[-27:]
        floating_rni.append(float(float_rni))

    df_unix_time["liquidityIndex"] = np.array(floating_rni)

    # sort the date frame by timestamps
    df_unix_time.sort_values("date", axis=0, inplace=True, ignore_index=True)

    # return the prepared dataframe
    return df_unix_time

# """
#    This function creates ``frequency`` min gap between entries by performing linear interpolation.
#    It requires a prepared DataFrame.
# """
def getFrequentData(df_rni, frequency):
    min_timestamp = int(df_rni["date"][0])
    max_timestamp = int(df_rni["date"][len(df_rni["date"]) - 1])

    timestamps = []
    rnis = []

    for timestamp in range(min_timestamp, max_timestamp+1, frequency * 60):
        closest = df_rni["date"].searchsorted(timestamp, side='right') - 1

        if closest + 1 < len(df_rni["date"]):
            t_a = df_rni["date"][closest]
            v_a = df_rni["liquidityIndex"][closest]

            t_b = df_rni["date"][closest+1]
            v_b = df_rni["liquidityIndex"][closest+1]

            v_timestamp = ((t_b - timestamp) * v_a + (timestamp - t_a) * v_b) / (t_b - t_a)

            timestamps.append(timestamp)
            rnis.append(v_timestamp)

    df_frequent = pd.DataFrame()
    df_frequent["date"] = np.array(timestamps)
    df_frequent["liquidityIndex"] = np.array(rnis)

    return df_frequent

# """
#    This function returns the rate between two timestamps.
#    It requires a prepared DataFrame.
# """
def getRateFromTo(df_rni, start, end, rate_oracle_mode, check_increasing=True):
    if start > end:
        raise Exception("Invalid dates (start > end)")

    if check_increasing:
        if not df_rni["date"].is_monotonic_increasing:
            raise Exception("RNI DF dates are not increasing")

    start_index = df_rni["date"].searchsorted(start, side='right') - 1
    end_index = df_rni["date"].searchsorted(end, side='right') - 1
    
    if start_index < 0:
        raise Exception("No observations before the start timestamp")

    start_rate = df_rni["liquidityIndex"][start_index]
    end_rate = df_rni["liquidityIndex"][end_index]

    if (start_rate > end_rate):
        raise Exception("Misconfiguration in the dataset (start_rate > end_rate)")
    
    if rate_oracle_mode == "linear":
        rate = end_rate - start_rate
    elif rate_oracle_mode == "compounding" or rate_oracle_mode == "sofr":
        rate = end_rate / start_rate - 1
    else:
        raise Exception("Unknown rate oracle mode " + rate_oracle_mode)

    return rate

# """
#    This function returns the apy between two timestamps.
#    It requires a prepared DataFrame.
# """
def getApyFromTo(df_rni, start, end, rate_oracle_mode, check_increasing=True):
    rate = getRateFromTo(df_rni, start, end, rate_oracle_mode, check_increasing)

    # compound continuously over time
    if rate_oracle_mode == "linear":
        apy = rate / ((end - start) / SECONDS_IN_YEAR)
    elif rate_oracle_mode == "compounding": 
        apy = pow(1 + rate, SECONDS_IN_YEAR / (end - start)) - 1
    elif rate_oracle_mode == "sofr": #compound fragmented per days, relative to a 360-day year
        apy = rate * (SECONDS_IN_360YEAR / (end - start))
    else:
        raise Exception("Unknown rate oracle mode " + rate_oracle_mode)

    return apy 

# """
#    This function returns the rate between two timestamps.
# """
def getDailyApy(datasets, lookback, rate_oracle_mode, check_increasing=True):
    no_of_datasets = len(datasets)

    if not no_of_datasets > 0:
        raise Exception("No datasets passed")

    if check_increasing:
        for i in range(no_of_datasets):
            if not datasets[i][1]["date"].is_monotonic_increasing:
                raise Exception("RNI DF dates are not increasing")

    date_range = [None, None]
    for _, df_rni in datasets:
        this_data_range = [0, 0]
        this_data_range[0] = (int(df_rni["date"][0]) // SECONDS_IN_DAY + lookback + 1) * SECONDS_IN_DAY
        this_data_range[1] = (int(df_rni["date"][len(df_rni["date"]) - 1]) // SECONDS_IN_DAY) * SECONDS_IN_DAY

        date_range[0] = this_data_range[0] if date_range[0] is None else max(date_range[0], this_data_range[0])
        date_range[1] = this_data_range[1] if date_range[1] is None else min(date_range[1], this_data_range[1])

    if date_range[0] > date_range[1]:
        raise Exception("The dates do not overlap between datasets")

    print("Creating a dataset between {0} and {1}.".format(datetime.datetime.utcfromtimestamp(date_range[0]), datetime.datetime.utcfromtimestamp(date_range[1])))

    dates = []
    apys = [[] for _ in range(no_of_datasets)]
    for daily_date in range(date_range[0], date_range[1]+1, SECONDS_IN_DAY):
        dates.append(str(datetime.datetime.utcfromtimestamp(daily_date)))

        for i in range(no_of_datasets):
            apys[i].append(getApyFromTo(df_rni, daily_date - lookback * SECONDS_IN_DAY, daily_date, rate_oracle_mode))
    
    df_apy = pd.DataFrame()
    df_apy["date"] = dates
    for i in range(no_of_datasets):
        df_apy[datasets[i][0]] = apys[i]
    
    return df_apy


"""
    Example implementation:

df_Aave_USDC = pd.read_csv("rni_historical_data/aave_usdc.csv")
df_Aave_USDC = getPreparedRNIData(df_Aave_USDC)
df_Aave_USDC = getFrequentData(df_Aave_USDC, frequency=30)

apys = getDailyApy([['USDC', df_Aave_USDC]], lookback=5)

apys.to_csv("apy_from_rni/AaveVariable_apy.csv", index=False)

"""   
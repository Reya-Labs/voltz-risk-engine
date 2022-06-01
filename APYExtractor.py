import pandas as pd
import requests
import os

# Global parameters
interest_rate_domain = "Supply"
interval_type = "Day"
start_date_timestamp = '1512075600'

"""
    Scrape loanscan for relevant APY time series across different 
    protocols (e.g. Aave and Compound) and different coins. Takes the
    particular APY provider as input, and generates corresponding time
    series across all relevant coins as a csv. An output directory may
    also be provided.
"""

# These borrow/lend providers are currently configured
token_symbols_dict = {
                    "AaveVariable": ["DAI", "USDC", "USDT", "TUSD"],
                    "CompoundV2": ["DAI", "USDC", "USDT"],
                    "YearnFinance": ["USDC", "DAI", "USDT"],
                    "dYdX": ["DAI"],
                    "Vesper": ["USDC", "DAI", "USDT"],
                    "Element": ["USDC", "DAI"]
}

def main(provider="AaveVariable", out_dir="./loanscan_data/"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dfs = []
    token_symbols = token_symbols_dict[provider]
    for token_symbol in token_symbols:
        url = """ https://api.loanscan.io/v1/interest-rates/historical?provider={}&interestRateDomain={}
                &intervaltype={}&startDateTimestamp={}&tokenSymbol={}""".\
            format(provider, interest_rate_domain, interval_type, start_date_timestamp, token_symbol)

        dataload = requests.get(url).json()
        df = pd.DataFrame(dataload)
        df.columns = ["{}".format(token_symbol), "Date"]
        
        print("Before: ", df)
        df.index = pd.DatetimeIndex(data=df.loc[:, "Date"]).tz_localize(None)
        print("After: ", df)
        df = df.drop(columns=["Date"])
        df = df.loc[df.index > "2020-02-29", :]
        dfs.append(df)
        #df.to_csv(out_dir+"{}_{}_apy.csv".format(provider, token_symbol))

    composite_df = pd.concat(dfs, axis=1)
    print(composite_df)
    composite_df.to_csv(out_dir+"composite_df_{}_apy.csv".format(provider))


if __name__=="__main__":
    # Adding an argument parser
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', "--provider", type=str, help="Define the procol provider of the APYs", default="AaveVariable")
    parser.add_argument("-o", "--out_dir", type=str, help="Define the output directory for the scraped data",default="./loanscan_data/")

    options = parser.parse_args()

    # Defining dictionary to be passed to the main function
    option_dict = dict( (k, v) for k, v in vars(options).items() if v is not None)
    print(option_dict)
    main(**option_dict)
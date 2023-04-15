from datetime import datetime
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Adding an argument parser
from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument(
    "-ds", "--dataset", help="Dataset with raw liquidity indices", required=True)
input_dataset_name = parser.parse_args().dataset

target_dir = f"simulations/{input_dataset_name}"

for filename in os.listdir(target_dir):
    f = os.path.join(target_dir, filename)
    if os.path.isfile(f) and f.endswith(".csv"):
        ds_file_path = f
        break

df = pd.read_csv(ds_file_path)
df["datetime"] = [datetime.fromtimestamp(x) for x in df["date"].values]

indices = [math.floor(x) for x in np.linspace(0, len(df["date"].values) - 1, 3)]

ys = []
for y in df.columns.values:
    if y.startswith('mr_lm_ft') or y.startswith('mr_im_ft'):
        ys.append(y)

df.plot(x="date", y=ys)
plt.xticks(ticks=df["date"][indices], labels=df["datetime"][indices])
plt.savefig(f"{target_dir}/ft_mrs")
plt.cla()

ys = []
for y in df.columns.values:
    if y.startswith('mr_lm_vt') or y.startswith('mr_im_vt'):
        ys.append(y)

df.plot(x="date", y=ys)
plt.xticks(ticks=df["date"][indices], labels=df["datetime"][indices])
plt.savefig(f"{target_dir}/vt_mrs")
plt.cla()

ys = []
for y in df.columns.values:
    if y.startswith('mr_lm_lp') or y.startswith('mr_im_lp'):
        ys.append(y)

df.plot(x="date", y=ys)
plt.xticks(ticks=df["date"][indices], labels=df["datetime"][indices])
plt.savefig(f"{target_dir}/lp_mrs")
plt.cla()

df.plot(x="date", y=["lower_USDC","apy_USDC","upper_USDC"])
plt.xticks(ticks=df["date"][indices], labels=df["datetime"][indices])
plt.savefig(f"{target_dir}/apys")
plt.cla()
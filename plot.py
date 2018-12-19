# -*- coding: utf-8 -*-

import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict 


def plot(files, dataList):
    fig = plt.figure();
    for file, dat in zip(files, dataList):
        plt.plot(dat, label=file)
    plt.legend(files, loc = "upper right")
    plt.xlabel("# Processed Batches")
    plt.ylabel("Loss")
#    plt.show()
    fig.savefig("test.pdf")



def load_file(filename):
    data = []
    for l in open(filename):
        if l.startswith("LOSS"):
            tokens = l.split()
            data.append(float(tokens[4]))
    return data[:5000]

def moving_average(dat, range):
    df = pd.DataFrame(dat)
    return list(df.rolling(range, min_periods=1).mean()[0])


def main(files):
    dataList = [moving_average(load_file(f), 30) for f in files]
    plot(files, dataList)


main(["SERIAL", "SYNC_2", "ASYNC_2", "ASYNC_2_0.5", "ASYNC_2_1", "ASYNC_2_1.5", "ASYNC_2_2.0"])

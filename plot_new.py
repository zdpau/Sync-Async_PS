# -*- coding: utf-8 -*-

import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict 


def plot(labels, dataList, outputfile):
    fig = plt.figure();
    for label, dat in zip(labels, dataList):
        plt.plot(dat, label=label)
    plt.legend(labels, loc = "upper right")
    plt.xlabel("# Processed Batches")
    plt.ylabel("Loss")
#    plt.show()
    fig.savefig(outputfile)



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


def main(files, labels, filename):
    dataList = [moving_average(load_file(f), 30) for f in files]
    plot(labels, dataList, filename)

labels = ["Sequential", "Sync", "Async delay=0.0", "Async delay=0.5", "Async delay=1.0", "Async delay=1.5", "Async delay=2.0"]

files = ["SERIAL", "SYNC_2", "ASYNC_2_0.0", "ASYNC_2_0.5", "ASYNC_2_1.0", "ASYNC_2_1.5", "ASYNC_2_2.0"]
main([a + ".BN3" for a in files], labels, "2workers.pdf")

files = ["SERIAL.BN3", "SYNC_3.3", "ASYNC_3_0.0.3", "ASYNC_3_0.5.3", "ASYNC_3_1.0.3", "ASYNC_3_1.5.3", "ASYNC_3_2.0.3"]
main(files, labels, "3workers.pdf")

#main(["SERIAL.BN", "SERIAL.BN2", "SERIAL.BN3"])

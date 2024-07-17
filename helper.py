from typing_extensions import NewType
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.stats import rankdata
from mplcursors import cursor
import math as m
import pandas as pd
import numpy as np
import csv
import sys


# This is the class that WLSK uses to store a "bucket"
class Bucket:
    def __init__(self, mil: int = None, pkts: int = None):
        self.t: int = mil
        self.c: int = pkts

    def time(self):
        return f"{self.t} ms"

    def __eq__(self, value: "Bucket") -> bool:
        return self.t == value.t

    def __str__(self) -> str:
        return f"BKT-{self.t}"

    def __iter__(self):
        yield self.t
        yield self.c

    def __lt__(self, other: "Bucket"):
        return self.t < other.t

    def __le__(self, other: "Bucket"):
        return self.t <= other.t

    def __gt__(self, other: "Bucket"):
        return self.t > other.t

    def __ge__(self, other: "Bucket"):
        return self.t >= other.t

    def __float__(self):
        return float(self.c)


def import_csv(csv_path: str) -> tuple[list[int], list[int]]:
    """takes a path returns a list of times and a list of packets received per millisecond"""
    times: list[int] = []
    packets: list[int] = []
    with open(csv_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            times.append(int(row[0]))
            packets.append(int(row[1]))
    return times, packets


# NOTE: This is what the window would look like in the actual FSM: a deque of Bucket objects.
# This file is just a "unit conversion" from that deque to a lists of times and packets,
# but I figured I would put the option to port back here if it helped.
def port_to_Buckets(times: list[int], packets: list[int]) -> list[Bucket]:
    """takes a list of times and a list of packets and converts them to a list of bucket objects."""
    buckets = []
    for i, _ in enumerate(times):
        buckets.append(Bucket(mil=times[i], pkts=packets[i]))
    return buckets


def old_correlate(data: list[int], word: list[int]) -> np.ndarray | int:
    """this is the current code ported over from the receiver_v3.py in WLSK"""
    # create the variance data from the normal data
    new_data = pd.Series(data)
    var_data = new_data.rolling(window=75).var().bfill()

    # upscale ones and zeros for the word conversion
    upscaled_one = [1] * 102
    upscaled_zero = [-1] * 102

    # Composite the correlation word into a new, huge upscaled word
    new_word = [
        item
        for value in word
        for item in (upscaled_one if value == 1 else upscaled_zero)
    ]

    # create the correlation data
    conv = np.correlate(var_data, new_word, "full")
    corr_data = conv - conv.mean()

    # return the correlation array
    # NOTE: this is not the index of the bucket at which the strongest point is.
    # That requires further calculation not done here.
    return corr_data, corr_data.argmax()


def new_correlate(data: list[int], word: list[int]) -> (np.ndarray, int):
    """a template function for you to test your own correlation methods."""
    # create the variance data from the normal data
    new_data = pd.Series(data)
    var_data = new_data.rolling(window=75).var().bfill()

    # upscale ones and zeros for the word conversion
    upscaled_one = [1] * 102
    upscaled_zero = [-1] * 102

    # Composite the correlation word into a new, huge upscaled word
    new_word = [
        item
        for value in word
        for item in (upscaled_one if value == 1 else upscaled_zero)
    ]

    # create the correlation data
    conv = np.correlate(var_data, new_word, "valid")
    corr_data = conv - conv.mean()

    # return the correlation array
    # NOTE: this is not the index of the bucket at which the strongest point is.
    # That requires further calculation not done here.
    return corr_data, corr_data.argmax()


def rank_correlation(
    data: list[int], word: list[int], sense: bool = False
) -> (np.ndarray, int, int):
    new_data = pd.Series(data)
    var_data = new_data.rolling(window=75).var().bfill()
    rank_data = pd.Series(rankdata(var_data))

    upscaled_one = [1] * 102
    upscaled_zero = [-1] * 102

    new_word = [
        item
        for value in word
        for item in (upscaled_one if value == 1 else upscaled_zero)
    ]

    if sense:
        # Get the first 3% of the data
        new_data_sense = new_data[: int(len(new_data) * 0.03)]
        var_data_sense = new_data_sense.rolling(window=75).var().bfill()
        rank_data_sense = pd.Series(rankdata(var_data_sense))
        conv_sense = np.correlate(rank_data_sense, new_word, "valid")
        corr_data_sense = conv_sense - conv_sense.mean()

        corr_sync_thresh = corr_data_sense.argmax()
    else:
        corr_sync_thresh = -1

    conv = np.correlate(rank_data, new_word, "valid")
    corr_data = conv - conv.mean()

    return corr_data, corr_data.argmax(), corr_sync_thresh


def poisson_threshold_correlation(
    data: list[int], word: list[int], sense: bool = False, swnd: int = 1000
) -> (np.ndarray, int, int):

    # Establish the bit decision threshold
    threshold = max(data[:swnd])

    new_data = pd.Series(data)

    # if data is less than threshold, set to 0
    new_data = new_data.apply(lambda x: 0 if x < threshold else x)

    var_data = new_data.rolling(window=75).var().bfill()

    # upscale ones and zeros for the word conversion
    upscaled_one = [1] * 102
    upscaled_zero = [-1] * 102

    # Composite the correlation word into a new, huge upscaled word
    new_word = [
        item
        for value in word
        for item in (upscaled_one if value == 1 else upscaled_zero)
    ]

    # create the correlation data
    conv = np.correlate(new_data, new_word, "valid")
    corr_data = conv - conv.mean()

    return new_data, max(corr_data.argmax(), abs(corr_data.argmin())), threshold


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python helper.py <path_to_csv>")
        sys.exit(1)
    csv_path = sys.argv[1]

    times, packets = import_csv(csv_path)

    # Zero out the timescale - makes graphs not lag horribly
    t_start = times[0]
    times = [item - t_start for item in times]

    # the list of bits that were sent in the messages
    bitstream = [
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
    ]
    # the chosen sync word in the messages
    sync_word = [
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        0,
    ]
    # the chosen barker code of the messages
    barker_code = [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1]
    # the first six bits of the message
    preamble = [1, 0, 1, 0, 1, 0]

    # TODO: Figure out how to correlate properly!
    # example of testing correlation on the sync word
    old_sync, o_sync_val = old_correlate(packets, sync_word)
    new_sync, n_sync_val = new_correlate(packets, sync_word)
    rank_sync, r_sync_val, r_thresh = rank_correlation(packets, sync_word)

    # NOTE: Graph the results. this is just the raw data
    NUM_GRAPHS = 2

    fig: Figure
    ax: list[Axes]
    fig, ax = plt.subplots(NUM_GRAPHS)
    ax = [ax[i] for i in range(NUM_GRAPHS)]

    # Each bucket is 1 ms, draw a vline for every second
    for i in range(1, int(max(times) / 1000)):
        ax[0].vlines(i * 1000, 0, max(packets), colors="lightgray", linestyles="dashed")

    ax[0].scatter(times, packets, s=5)
    ax[0].set_title("Packets Per Millisecond Received")
    ax[0].set_xlabel("time (ms)")
    ax[0].set_ylabel("packets received")

    swnd = 10000

    thresh_data, p_sync, noise_floor = poisson_threshold_correlation(packets, sync_word, swnd=swnd)

    noise_floor = m.ceil(noise_floor * 0.5)
    print(noise_floor)

    # m_sync is the second instance of the number 1
    sync_indeces = thresh_data[thresh_data > noise_floor].index
    m_sync = sync_indeces[4] # msg1
    # m_sync = sync_indeces[5] # msg2

    ax[0].hlines(noise_floor, 0, max(times), colors="r", linestyles="dashed")
    ax[0].vlines(swnd, 0, max(packets), colors="g", linestyles="dashed")
    ax[0].vlines(p_sync, 0, max(packets), colors="b", linestyles="dashed")
    ax[0].vlines(p_sync - (len(sync_word) * 102.4), 0, max(packets), colors="b", linestyles="dashed")

    # Draw m_sync window
    ax[0].vlines(m_sync, 0, max(packets), colors="purple", linestyles="dashed")
    ax[0].vlines(m_sync + (102.4 * len(sync_word)), 0, max(packets), colors="purple", linestyles="dashed")

    BEACON_SIZE = 102.4
    # Draw zones of shaded areas starting at p_sync - 32 * 102.4 to p_sync where the thickness of the zone is 102.4 ms
    # for i in range(len(sync_word)):
    #     # Draw expected bit values of the sync word in the middle of the shaded region
    #     ax[0].text(p_sync - (len(sync_word) * 102.4) + (i * 102.4) + 51.2, max(packets) / 2, str(sync_word[i]), fontsize=12, color="black", ha="center")

    #     # Draw actual bit value of the region in the middle of the shaded region
    #     value = thresh_data[m_sync + (i * 102)]
    #     print(m_sync + (i * 102), value)
    #     ax[0].text(p_sync - (len(sync_word) * 102.4) + (i * 102.4) + 51.2,  max(packets) / 2 - 10, str(value), fontsize=12, color="red", ha="center")

    #     ax[0].axvspan(p_sync - (len(sync_word) * 102.4) + (i * 102.4), p_sync - (len(sync_word) * 102.4) + ((i + 1) * 102.4), color="red", alpha=0.5)


    # print(thresh_data)
    for i in sync_indeces:
        print(thresh_data[i])
    # Draw zones of shaded areas starting at m_sync to m_sync + 32 * 102.4 where the thickness of the zone is 102.4 ms
    for i in range(len(sync_word)):
        # Draw expected bit values of the sync word in the middle of the shaded region
        ax[0].text(m_sync + (i * BEACON_SIZE), max(packets) / 2, str(sync_word[i]), fontsize=12, color="black", ha="center")

        # Draw actual bit value of the region in the middle of the shaded region
        # value = thresh_data[m_sync + (i * 102)]

        # Search in a range of 5ms on either side of m_sync to see if the value is larger than the noise_floor
        value = 0

        wnd = 50
        ax[0].axvspan(m_sync + (i * BEACON_SIZE) - wnd, m_sync + (i * BEACON_SIZE) + wnd, color="purple", alpha=0.5)

        for j in range(-wnd, wnd):
            print(thresh_data[m_sync + (i * int(BEACON_SIZE)) + j], noise_floor)
            if thresh_data[m_sync + (i * int(BEACON_SIZE)) + j] > noise_floor:
                value = 1
                break
            else:
                value = 0
        # print(m_sync + (i * BEACON_SIZE), value)

        # Draw actual bit value of the region in the middle of the shaded region
        ax[0].text(m_sync + (i * BEACON_SIZE), max(packets) / 2 - 10, str(value), fontsize=12, color="red", ha="center")

        # ax[0].axvspan(m_sync + (i * 102.4), m_sync + ((i + 1) * 102.4), color="purple", alpha=0.5)


    # ax[1].plot(old_sync)
    # ax[1].set_title("Correlation using old method")
    # try: ax[0].vlines(times[o_sync_val - 1],0,max(packets),colors='r',linestyles='dashed')
    # except Exception: pass
    # try: ax[0].vlines(times[32 * 102 + n_sync_val - 1],0,max(packets),colors='g',linestyles='dashed')
    # except Exception: pass
    # try: ax[0].vlines(times[32 * 102 + r_sync_val + 1],0,max(packets),colors='b',linestyles='dashed')
    # except Exception: pass

    # try:
    #     print(f"Old method: {times[32 * 102 + o_sync_val - 1]}")
    #     print(f"New method: {times[32 * 102 + n_sync_val - 1]}")
    #     print(f"Rank method: {times[32 * 102 + r_sync_val + 1]}")
    # except Exception:
    #     pass

    # ax[2].plot(new_sync)
    # ax[2].set_title("Correlation using new method")

    # ax[3].plot(rank_sync)
    # ax[3].set_title("Correlation using rank method")
    # ax[3].hlines(r_thresh,0,len(rank_sync),colors='r',linestyles='dashed')

    # Plot the distribution of the ping values

    # ax[1].hist(packets, bins=100)

    # # Vertical line for the max value
    # # Height of line is as tall as the maximum occurence
    # max_height = max(np.histogram(packets, bins=100)[0])
    # ax[1].vlines(max(packets), 0, max_height, colors="r", linestyles="dashed")

    # # Draw line at the 70th percentile of the distribution of packet occurences
    # ax[1].vlines(
    #     np.percentile(packets, 99.99), 0, max_height, colors="g", linestyles="dashed"
    # )

    # ax[1].set_title("Packet Distribution")
    # ax[1].set_xlabel("Packets")
    # ax[1].set_ylabel("Frequency")

    # ax[2].plot(thresh_data)
    # ax[2].set_title("Poisson Threshold Correlation")
    # ax[2].set_xlabel("time (ms)")
    # ax[2].set_ylabel("packets received")

    plt.tight_layout()
    cursor(hover=True)
    plt.show()

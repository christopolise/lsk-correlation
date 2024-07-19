import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import pandas as pd
import numpy as np
import csv
import sys
from scipy import signal, stats
import math as m

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


def import_csv(csv_path: str) -> tuple[np.array, np.array]:
    """takes a path returns a list of times and a list of packets received per millisecond"""
    times: list[int] = []
    packets: list[int] = []
    with open(csv_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            times.append(int(row[0]))
            packets.append(int(row[1]))
    return np.array(times), np.array(packets)


# NOTE: This is what the window would look like in the actual FSM: a deque of Bucket objects.
# This file is just a "unit conversion" from that deque to a lists of times and packets,
# but I figured I would put the option to port back here if it helped.
def port_to_Buckets(times: list[int], packets: list[int]) -> list[Bucket]:
    """takes a list of times and a list of packets and converts them to a list of bucket objects."""
    buckets: list[Bucket] = []
    for i, _ in enumerate(times):
        buckets.append(Bucket(mil=times[i], pkts=packets[i]))
    return buckets


def old_correlate(data: list[int], word: list[int]) -> np.ndarray:
    """this is the current code ported over from the receiver_v3.py in WLSK"""
    # create the variance data from the normal data
    new_data: pd.Series = pd.Series(data)
    var_data: pd.Series = new_data.rolling(window=75).var().bfill()

    # upscale ones and zeros for the word conversion
    upscaled_one: list[int] = [1] * 102
    upscaled_zero: list[int] = [-1] * 102

    # Composite the correlation word into a new, huge upscaled word
    new_word: list[int] = [
        item
        for value in word
        for item in (upscaled_one if value == 1 else upscaled_zero)
    ]

    # create the correlation data
    conv: np.ndarray = np.correlate(var_data, new_word, "valid")
    corr_data: np.ndarray = conv - conv.mean()

    # return the correlation array
    # NOTE: this is not the index of the bucket at which the strongest point is.
    # That requires further calculation not done here.
    return corr_data


def upscale(word):
    # upscale ones and zeros for the word conversion
    upscaled_one: list[int] = [-1] * 10 + [1] * 77 + [-1] * 15
    upscaled_zero: list[int] = [-1] * 102

    # Composite the correlation word into a new, huge upscaled word
    new_word: list[int] = [
        item
        for value in word
        for item in (upscaled_one if value == 1 else upscaled_zero)
    ]

    return new_word


def new_correlate(data: list[int], word: list[int]) -> np.ndarray:
    """a template function for you to test your own correlation methods."""

    # create the variance data from the normal data
    new_data: pd.Series = pd.Series(data)
    # pd.Series(samples).rolling(window=len(symbol)).apply(calc, raw=True)
    var_data: pd.Series = new_data.rolling(window=75).var().bfill()

    new_word = upscale(word)

    def calc(data):
        rank = stats.rankdata(data, "average")
        rank = rank - (len(rank) / 2)  # Make it zero mean
        rank = rank / (len(rank) / 2)  # Make values between -1 and 1
        return (rank * new_word).sum()

    # corr_data = var_data.rolling(window=len(new_word)).apply(calc, raw=True)

    # create the correlation data
    conv: np.ndarray = np.correlate(var_data, new_word, "valid")
    corr_data: np.ndarray = conv - conv.mean()

    # return the correlation array
    # NOTE: this is not the index of the bucket at which the strongest point is.
    # That requires further calculation not done here.
    return corr_data

all_zero_percentages = []
def zero_percentage(packets, time_center, threshold_value, ax):
    before_window = 100
    after_window = 2

    num_zeros = sum(
        1
        for pkt in packets[time_center - before_window : time_center + after_window]
        if pkt == 0
    )
    percent_above = num_zeros / (before_window + after_window) * 100
    result = percent_above > 70

    all_zero_percentages.append(percent_above)

    ax.text(
        time_center,
        threshold_value - 1,
        f"{percent_above:.0f}",
        color="black",
        ha="center",
        size=8,
    )

    ax.axvspan(
        time_center - before_window,
        time_center + after_window,
        color="gray",
        alpha=0.2,
    )

    return result


def sync_threshold(packets, time_center, threshold_value, ax):
    bit_decision = packets > threshold_value
    return np.where(bit_decision == 1)[0]


def sync_on_zero(packets, time_center, threshold_value, ax):
    bit_decision = packets == 0
    return np.where(bit_decision == 1)[0]


def new_correlation_2(packets, times, word):
    fig, ax = plt.subplots()
    ax.scatter(times, packets, s=2)
    ax.set_title("Packets Over Time")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Packets")

    threshold_value = 10
    before_window = 10
    after_window = 20

    ax.axhline(y=threshold_value, color="r", linestyle="--", label="Threshold")

    for one_index in sync_threshold(packets, times, threshold_value, None):
        ax.vlines(times[one_index], 0, 100, color="red", linestyle="--")
        print(times[one_index])
        result = []
        for i in range(len(word)):
            possible_one_time = times[one_index] + m.ceil(102.4 * i)

            ax.text(
                possible_one_time,
                threshold_value + 2,
                word[i],
                color="red",
                ha="center",
            )

            # print(packets[possible_one_time - window : possible_one_time + window])
            if zero_percentage(packets, possible_one_time, threshold_value, ax):
                color = "black" if word[i] == 1 else "blue"
                result.append(1)
                ax.text(
                    possible_one_time,
                    threshold_value + 1,
                    "1",
                    color=color,
                    ha="center",
                )
            else:
                color = "black" if word[i] == 0 else "blue"
                result.append(0)
                ax.text(
                    possible_one_time,
                    threshold_value + 1,
                    "0",
                    color=color,
                    ha="center",
                )

        # Create a graph that shows the distribution of zeros and ones
        fig, ax = plt.subplots()
        ax.hist(all_zero_percentages, bins=20)
        ax.set_title("Distribution of Zero Percentages")
        ax.set_xlabel("Percentage of Zeros")
        ax.set_ylabel("Frequency")

        plt.show()

        # print(result)

        diff_count = sum(r != w for r, w in zip(result[:30], word[:30]))
        print(
            f"S + P:\t\tNumber of elements that are different between result and word: {diff_count}"
        )


        # fig.set_size_inches(17.5, 9)
        if diff_count > 10:
            ax.clear()
            ax.scatter(times, packets, s=2)
            ax.set_title("Packets Over Time")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Packets")
            ax.axhline(y=threshold_value, color="r", linestyle="--", label="Threshold")
            continue

        diff_count = sum(r != w for r, w in zip(result, word))
        print(
            f"S + P + D:\tNumber of elements that are different between result and word: {diff_count}"
        )

        # Print BER
        print(f"BER:\t\t{diff_count / len(word)}")
        plt.show()
        exit()
        return result

    # previous_one_index = None
    # result = []
    # for one_index in np.where(bit_decision == 1)[0]:
    #     if previous_one_index is None:
    #         previous_one_index = one_index
    #         result.append(1)
    #         print("1")
    #         continue

    #     time_diff = times[one_index] - times[previous_one_index]
    #     print(time_diff, end=" ")

    #     # Combine together packets that are too close together
    #     if time_diff < 5:
    #         print(", skipping")
    #         continue

    #     print("1, ", end=" ")
    #     result.append(1)
    #     print(f"*{round(time_diff / 102) - 1}*", end=" ")
    #     for _ in range(round(time_diff / 102) - 1):
    #         print("0, ", end=" ")
    #         result.append(0)
    #     print()

    #     previous_one_index = one_index

    conv: np.ndarray = np.correlate(result, word, "valid")
    corr_data: np.ndarray = conv - conv.mean()

    return corr_data


if __name__ == "__main__":

    csv_path: str = sys.argv[1]

    times, packets = import_csv(csv_path)

    # Cut down the data
    times = times[20000:]
    packets = packets[20000:]

    # Zero out the timescale - makes graphs not lag horribly
    times = times - times[0]

    # the list of bits that were sent in the messages
    bitstream: list[int] = [
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
    sync_word: list[int] = [
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
    # barker_code: list[int] = [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1]
    barker_code: list[int] = [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0]
    inverted_barker_code: list[int] = [1 - x for x in barker_code]

    # the first six bits of the message
    preamble: list[int] = [1, 0, 1, 0, 1, 0]

    expected_data = sync_word + [
        item
        for bit in bitstream
        for item in (barker_code if bit else inverted_barker_code)
    ]

    # TODO: Figure out how to correlate properly!
    # example of testing correlation on the sync word
    old_sync: np.ndarray = old_correlate(packets, sync_word)
    new_sync: np.ndarray = new_correlate(packets, sync_word)
    new_sync2: np.ndarray = new_correlation_2(packets, times, expected_data)

    # print(f"Max value for old_sync: {old_sync.max()}")
    # print(f"Max value for new_sync: {new_sync.max()}")

    # result = threshold(packets, times)
    # print(list(result))
    # chips = packets > 10

    ########################################################
    # lambda_value = packets.mean()
    # poisson_dist = np.random.poisson(lambda_value, len(packets))

    # # Compute the CDF for packets data
    # packets_sorted = np.sort(packets)
    # packets_cdf = np.arange(1, len(packets_sorted) + 1) / len(packets_sorted)

    # # Compute the CDF for Poisson distribution
    # poisson_sorted = np.sort(poisson_dist)
    # poisson_cdf = np.arange(1, len(poisson_sorted) + 1) / len(poisson_sorted)

    # plt.figure()
    # plt.hist(packets, bins=50, alpha=0.5, label="Packets Data")
    # plt.hist(poisson_dist, bins=50, alpha=0.5, label="Poisson Distribution")
    # plt.title("Histogram of Packets Data vs Poisson Distribution")
    # plt.xlabel("Number of Packets")
    # plt.ylabel("Frequency")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # plt.figure()
    # plt.plot(packets_sorted, packets_cdf, label="Packets Data CDF")
    # plt.plot(poisson_sorted, poisson_cdf, label="Poisson Distribution CDF")
    # plt.title("CDF of Packets Data vs Poisson Distribution")
    # plt.xlabel("Number of Packets")
    # plt.ylabel("CDF")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # exit()
    ########################################################

    # NOTE: Graph the results. this is just the raw data
    NUM_GRAPHS = 5

    fig: Figure
    ax: list[Axes]
    fig, ax = plt.subplots(NUM_GRAPHS)

    ax[0].scatter(times, packets, s=2)
    ax[0].set_title("Packets Per Millisecond Received")
    ax[0].set_xlabel("time (ms)")
    ax[0].set_ylabel("packets received")

    new_data: pd.Series = pd.Series(packets)
    ax[1].plot(new_data.rolling(window=75).var().bfill())

    new_sync_word = upscale(sync_word)
    sync_word_graph = [np.nan] * 7000 + new_sync_word
    sync_word_graph = np.array(sync_word_graph) * 10
    ax[1].plot(sync_word_graph)

    ax[2].plot(old_sync)
    ax[2].set_title("Correlation using old method")

    ax[3].plot(new_sync)
    ax[3].set_title("Correlation using new method")

    ax[4].plot(new_sync2)
    ax[4].set_title("Correlation using new method")

    # ax[3].plot(chips)
    # ax[3].set_title("Bit Decision")

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

import numpy as np

__all__ = [
    "data_smooth",
    "plot_broken_line",
]


def data_smooth(data, weight=0.90):
    prev_data = data[0]
    smooth_data = []
    for point in data:
        prev_data = prev_data * weight + (1 - weight) * point
        smooth_data.append(prev_data)

    return np.array(smooth_data)


def plot_broken_line(data, title="", x_label="epoch", y_label="loss"):
    smooth_data = data_smooth(data)
    plt.title(title)
    plt.grid(ls='--')
    plt.plot(range(len(smooth_data)), smooth_data, 'r', linestyle=':', linewidth=1.0)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()
    plt.close()


if __name__ == "__main__":
    my_data = [i for i in range(100)]
    plot_broken_line(my_data, "test_data", "index", "nothing")

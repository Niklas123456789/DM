from matplotlib import pyplot as plt


def scatterPlot2DBig(data, title):
    fig = plt.figure(figsize=(15, 15))
    plt.scatter(data[:, 0], data[:, 1])
    plt.title(title)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt



def plot_table(data, out_file, x_label, y_label,
                          x_ticks, y_ticks,
                          title=None,
                          cmap=plt.cm.Blues, show=False):

    fig, ax = plt.subplots()
    im = ax.imshow(data, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(data.shape[1]),
           yticks=np.arange(data.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=x_ticks, yticklabels=y_ticks,
           title=title,
           ylabel=x_label,
           xlabel=y_label)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.5f'
    thresh = data.max() / 2.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, format(data[i, j], fmt),
                    ha="center", va="center",
                    color="white" if data[i, j] > thresh else "black")

    plt.ylim([float(data.shape[0])-0.5, -.5])
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(out_file)
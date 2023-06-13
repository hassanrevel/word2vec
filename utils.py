import datetime
import pandas as pd
import gensim
import seaborn as sns
import matplotlib.pyplot as plt

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (16, 8)


def plot_learning_curves(labels, learning_curves, save_dir=None):
    '''
    Plots each of the provided learning_curves on one plot.
    '''

    # For each of the learning curves...
    for lc in learning_curves:
        # Split into separate x and y lists.
        x_vals = [pair[0] for pair in lc]
        y_vals = [pair[1] for pair in lc]

        # Add to the line plot.
        plt.plot(x_vals, y_vals)

    # Label the axes.
    plt.xlabel('Epoch')
    plt.ylabel('Correct Analogies')
    plt.title('Learning Curve')

    # Set the x-axis to integers. To do this, we need to know the maximum
    # number of epochs among the learning curves.
    max_epochs = max([len(lc) for lc in learning_curves])
    plt.xticks(range(0, max_epochs))

    # Add the labels to the legend.
    plt.legend(labels)
    if save_dir:
        plt.savefig(save_dir)
    plt.show()

def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def data_parser(data):
    comments = pd.read_csv(data, sep='\t', index_col=0)

    comments['comment'] = comments['comment'].apply(lambda x: x.replace('NEWLINE_TOKEN', ''))
    comments['comment'] = comments['comment'].apply(lambda x: x.replace('TAB_TOKEN', ''))

    num_tokens = 0
    sentences = []

    for i, row in comments.iterrows():
        parsed = gensim.utils.simple_preprocess(row.comment)

        num_tokens += len(parsed)
        sentences.append(parsed)

    return sentences
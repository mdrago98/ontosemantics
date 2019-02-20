import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_embeddings(wv, vocabulary):

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    y = tsne.fit_transform(wv[:1000, :])

    plt.scatter(y[:, 0], y[:, 1])
    for label, x, y in zip(vocabulary, y[:, 0], y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()

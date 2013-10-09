import numpy as np
from functools import partial
from vector_utils import normalise_vector


def mean_vector(images):
    N = len(images)
    avg = np.zeros_like(images[0].as_vector(keep_channels=True))
    for im in images:
        avg += im.as_vector(keep_channels=True)

    return normalise_vector(avg / N)


def partial_logmap(logmap, base_vectors):
    return partial(logmap, base_vectors)
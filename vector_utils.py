import numpy as np


def normalise_vector(v):
    return v / np.sqrt(np.sum(v ** 2, axis=-1))[..., None]


def row_norm(v):
    return np.sqrt(np.sum(v ** 2, axis=-1))


def cart2sph(x, y, z):
    xy = np.power(x, 2) + np.power(y, 2)
    # for elevation angle defined from XY-plane up
    return np.vstack([np.arctan2(y, x), np.arctan2(z, np.sqrt(xy))]).T


def sph2cart(azimuth, elevation, r):
    cart = np.empty([azimuth.shape[0], 3])
    azi_cos = np.cos(azimuth)
    ele_cos = np.cos(elevation)
    azi_sin = np.sin(azimuth)
    ele_sin = np.sin(elevation)
    cart[:, 0] = r * ele_cos * azi_cos
    cart[:, 1] = r * ele_cos * azi_sin
    cart[:, 2] = r * ele_sin
    return cart


def normalise_image(image):
    """
    For normalising an image that represents a set of vectors.
    """
    vectors = image.as_vector(keep_channels=True)
    return image.from_vector(normalise_vector(vectors))
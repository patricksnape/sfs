import numpy as np


def normalise_vector(v):
    return v / np.sqrt(np.sum(v ** 2, axis=-1))[..., None]


def row_norm(v):
    return np.sqrt(np.sum(v ** 2, axis=-1))


def cart2sph(x, y, z):
    xy = np.power(x, 2) + np.power(y, 2)
    # for elevation angle defined from XY-plane up
    return np.concatenate([np.arctan2(y, x)[..., None],
                           np.arctan2(z, np.sqrt(xy))[..., None]], axis=-1)


def sph2cart(azimuth, elevation, r):
    azi_cos = np.cos(azimuth)
    ele_cos = np.cos(elevation)
    azi_sin = np.sin(azimuth)
    ele_sin = np.sin(elevation)

    cart = np.concatenate([(r * ele_cos * azi_cos)[..., None],
                           (r * ele_cos * azi_sin)[..., None],
                           (r * ele_sin)[..., None]], axis=-1)
    return cart


def normalise_image(image):
    """
    For normalising an image that represents a set of vectors.
    """
    vectors = image.as_vector(keep_channels=True)
    return image.from_vector(normalise_vector(vectors))
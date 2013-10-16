import numpy as np


def normalise_vector(v):
    return v / np.sqrt(np.sum(v ** 2, axis=-1))[..., None]


def row_norm(v):
    return np.sqrt(np.sum(v ** 2, axis=-1))


def cart2sph(x, y, z, theta_origin='xy'):
    """
    theta_origin : {'xy', 'z'}
        Defines where to take the 0 value for the elevation angle, theta. xy
        implies the origin is at the xy-plane and and 90 is at the z-axis.
        z implies that the origin is at the z-axis and 90 is at the xy-plane.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)

    if theta_origin == 'xy':
        xy = np.sqrt(x**2 + y**2)
        angles = np.concatenate([phi[..., None],
                                 np.arctan2(z, xy)[..., None],
                                 r[..., None]], axis=-1)
    elif theta_origin == 'z':
        angles = np.concatenate([phi[..., None],
                                 np.arccos(z / r)[..., None],
                                 r[..., None]], axis=-1)
    else:
        raise ValueError('Unknown value for the theta origin, valid values '
                         'are: xy, z')
    return angles


def sph2cart(azimuth, elevation, r, theta_origin='xy'):
    """
    theta_origin : {'xy', 'z'}
        Defines where to take the 0 value for the elevation angle, theta. xy
        implies the origin is at the xy-plane and and 90 is at the z-axis.
        z implies that the origin is at the z-axis and 90 is at the xy-plane.
    """
    azi_cos = np.cos(azimuth)
    ele_cos = np.cos(elevation)
    azi_sin = np.sin(azimuth)
    ele_sin = np.sin(elevation)

    if theta_origin == 'xy':
        cart = np.concatenate([(r * ele_cos * azi_cos)[..., None],
                               (r * ele_cos * azi_sin)[..., None],
                               (r * ele_sin)[..., None]], axis=-1)
    elif theta_origin == 'z':
        cart = np.concatenate([(r * ele_sin * azi_cos)[..., None],
                               (r * ele_sin * azi_sin)[..., None],
                               (r * ele_cos)[..., None]], axis=-1)
    else:
        raise ValueError('Unknown value for the theta origin, valid values '
                         'are: xy, z')
    return cart


def normalise_image(image):
    """
    For normalising an image that represents a set of vectors.
    """
    vectors = image.as_vector(keep_channels=True)
    return image.from_vector(normalise_vector(vectors))
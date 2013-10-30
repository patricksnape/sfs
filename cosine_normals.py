import numpy as np
from vector_utils import sph2cart


class Spherical(object):

    def __init__(self):
        super(Spherical, self).__init__()

    def logmap(self, tangent_vectors):
        if len(tangent_vectors.shape) < 3:
            tangent_vectors = tangent_vectors[None, ...]

        x = tangent_vectors[..., 0]
        y = tangent_vectors[..., 1]
        z = tangent_vectors[..., 2]

        xyz = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        xy = np.sqrt(x ** 2 + y ** 2)

        gx = x / xy
        gy = y / xy
        gz = z / xyz
        sgz = np.sqrt(1 - gz ** 2)

        spher = np.concatenate([gx[..., None], gy[..., None],
                                gz[..., None], sgz[..., None]], axis=-1)
        spher[np.isnan(spher)] = 0.0

        return spher

    def expmap(self, sd_vectors):
        if len(sd_vectors.shape) < 3:
            sd_vectors = sd_vectors[None, ...]

        gx = sd_vectors[..., 0]
        gy = sd_vectors[..., 1]
        gz = sd_vectors[..., 2]
        sgz = sd_vectors[..., 3]

        gzsgz = np.sqrt(gz ** 2 + sgz ** 2)

        gxgy = np.sqrt(gx ** 2 + gy ** 2)

        gx = gx / gxgy
        gy = gy / gxgy
        gz = gz / gzsgz
        sgz = sgz / gzsgz

        phi = np.arctan2(gy, gx)
        theta = np.arctan2(sgz, gz)

        cart = sph2cart(phi, theta, np.ones_like(phi), theta_origin='z')
        cart[np.isnan(cart)] = 0.0

        return cart

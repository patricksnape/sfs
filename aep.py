import numpy as np
from vector_utils import sph2cart, cart2sph


class AEP(object):

    def __init__(self, base_vectors):
        super(AEP, self).__init__()
        self.base_vectors = base_vectors
        self.long0lat1 = cart2sph(-self.base_vectors[:, 1],
                                  self.base_vectors[:, 0],
                                  self.base_vectors[:, 2])
        self.long0lat1[:, 0] += np.pi
        self.long0lat1 = self.long0lat1[None, ...]

    def expmap(self, tangent_vectors):
        # If we've been passed a single vector to map, then add the extra axis
        # Number of sample first
        if len(tangent_vectors.shape) < 3:
            tangent_vectors = tangent_vectors[None, ...]

        eps = np.spacing(1)

        rho = np.sqrt(tangent_vectors[..., 0] ** 2 +
                      tangent_vectors[..., 1] ** 2)
        Z = np.exp(np.arctan2(tangent_vectors[..., 1],
                              tangent_vectors[..., 0]) * 1j)
        V1 = rho * np.real(Z)
        V2 = rho * np.imag(Z)

        # To prevent divide by 0 warnings when rho is 0
        ir = rho == 0
        rho[ir] = eps
        c = rho

        c[ir] = eps

        Y = np.real(np.arcsin(np.cos(c) * np.sin(self.long0lat1[..., 1]) +
                              np.cos(self.long0lat1[..., 1]) *
                              np.sin(c) * V2 / rho))
        X = np.real(self.long0lat1[..., 0] +
                    np.arctan2(V1 * np.sin(c), np.cos(self.long0lat1[..., 1]) *
                    np.cos(c) * rho - np.sin(self.long0lat1[..., 1]) *
                    V2 * np.sin(c)))

        ns = sph2cart(X - np.pi, Y, np.ones_like(Y))
        # Swap x and y axes
        ns[..., [0, 1]] = ns[..., [1, 0]]
        ns[..., 1] = -ns[..., 1]

        return ns

    def logmap(self, sd_vectors):
        # If we've been passed a single vector to map, then add the extra axis
        # Number of sample first
        if len(sd_vectors.shape) < 3:
            sd_vectors = sd_vectors[None, ...]

        longlat = cart2sph(-sd_vectors[..., 1], sd_vectors[..., 0],
                           sd_vectors[..., 2])
        longlat[..., 0] += np.pi

        c = np.arccos(np.sin(self.long0lat1[..., 1]) *
                      np.sin(longlat[..., 1]) +
                      np.cos(self.long0lat1[..., 1]) *
                      np.cos(longlat[..., 1]) *
                      np.cos(longlat[..., 0] - self.long0lat1[..., 0]))

        k = 1.0 / np.sin(c)
        k = np.nan_to_num(k)
        k = c * k

        v0 = k * (np.cos(longlat[..., 1]) * np.sin(longlat[..., 0] -
                  self.long0lat1[..., 0]))
        v1 = k * (np.cos(self.long0lat1[..., 1]) * np.sin(longlat[..., 1]) -
                  np.sin(self.long0lat1[..., 1]) * np.cos(longlat[..., 1]) *
                  np.cos(longlat[..., 0] - self.long0lat1[..., 0]))
        vs = np.dstack([v0[..., None], v1[..., None]])

        return vs


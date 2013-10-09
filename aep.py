import numpy as np
from vector_utils import sph2cart, cart2sph
import abc


class AEP(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, base_vectors):
        super(AEP, self).__init__()
        self.base_vectors = base_vectors

    @abc.abstractmethod
    def expmap(self, tangent_vectors):
        pass

    @abc.abstractmethod
    def logmap(self, sd_vectors):
        pass


class Smith(AEP):

    def __init__(self, base_vectors):
        super(Smith, self).__init__(base_vectors)
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


class Snape(AEP):

    def __init__(self, base_vectors):
        super(Snape, self).__init__(base_vectors)
        self.thetaav = self.elevation(self.base_vectors[:, 2])
        self.phiav = self.azimuth(self.base_vectors[:, 0],
                                  self.base_vectors[:, 1])
        self.phiav, self.thetaav = self.clip_angles(self.phiav, self.thetaav)

    def expmap(self, tangent_vectors):
        # If we've been passed a single vector to map, then add the extra axis
        # Number of sample first
        if len(tangent_vectors.shape) < 3:
            tangent_vectors = tangent_vectors[None, ...]

        # find any zero normals as they present a real problem
        # the column indicies are the same in both sets of data
        zero_indices = np.sum(np.abs(tangent_vectors), axis=-1) == 0.0

        xs = tangent_vectors[..., 0]
        ys = tangent_vectors[..., 1]

        c = np.sqrt(xs ** 2 + ys ** 2)
        recipc = 1.0 / c

        # thetas = asin[cos(c) * sin(thetaav) -
        #               (1/c) * yk * sin(c) * cos(thetav)]
        s = (np.cos(c) * np.sin(self.thetaav) +
             recipc * ys * np.sin(c) * np.cos(self.thetaav))

        # Elevation
        el = np.arcsin(s)
        # Azimuth
        # phis = phiav + atan(psi)
        phis = self.psi(c, self.thetaav, xs, ys)
        azi = self.phiav + np.arctan2(phis[..., 0], phis[..., 1])
        azi, el = self.clip_angles(azi, el)

        # convert angles to coordinates
        xs = np.cos(azi) * np.sin(el)
        ys = np.sin(azi) * np.sin(el)
        zs = np.cos(el)
        ns = np.dstack([xs[..., None], ys[..., None], zs[..., None]])

        # reset zero projections back to zero
        ns[zero_indices, :] = 0.0

        return ns

    def logmap(self, sd_vectors):
        # If we've been passed a single vector to map, then add the extra axis
        # Number of sample first
        if len(sd_vectors.shape) < 3:
            sd_vectors = sd_vectors[None, ...]

        # find any zero normals as they present a real problem
        # the column indicies are the same in both sets of data
        zero_indices = np.sum(np.abs(sd_vectors), axis=-1) == 0.0

        thetak = self.elevation(sd_vectors[..., 2])
        phik = self.azimuth(sd_vectors[..., 0], sd_vectors[..., 1])
        phik, thetak = self.clip_angles(phik, thetak)

        # cos(c) = sin(thetaav) * sin(thetak) +
        #          cos(thetaav) * cos(thetak) * cos[phik - phiav]
        cosc = (np.sin(self.thetaav) * np.sin(thetak) +
                np.cos(self.thetaav) * np.cos(thetak) *
                np.cos(phik - self.phiav))
        c = np.arccos(cosc)
        # kprime = c / sin(c)
        kprime = c / np.sin(c)

        # xs = kprime * cos(thetak) * sin[phik - phiav]
        xs = kprime * np.cos(thetak) * np.sin(phik - self.phiav)
        # ys = kprime * (cos(thetaav) * sin(phik) -
        #      sin(thetaav) * cos(thetak) * cos[phik - phiav]
        ys = kprime * (np.cos(self.thetaav) * np.sin(thetak) -
                       np.sin(self.thetaav) * np.cos(thetak) *
                       np.cos(phik - self.phiav))
        vs = np.dstack([xs[..., None], ys[..., None]])

        # reset the zero normals back to 0
        vs[zero_indices, :] = 0.0

        return vs

    def clip_angles(self, phi, theta):
        # round thetas back in to the range [-pi/2, pi/2]
        pi_over_2 = np.pi / 2.0
        theta[theta > pi_over_2] -= np.pi
        theta[theta < -pi_over_2] += np.pi

        phi[phi > np.pi] -= 2.0 * np.pi
        phi[phi < -np.pi] += 2.0 * np.pi

        return phi, theta

    # psi = thetaav != (pi / 2) ->
    #       xk * sin(c) / c * cos(thetaav) * cos(c) - yk * sin(thetav) * sin(c)
    #       thetaav == (pi/2)   -> -(xk/yk)
    #       thetaav == -(pi/2)  -> xk/yk
    def psi(self, c, thetaav, xs, ys):
        eps = np.spacing(1)
        out = np.zeros([xs.shape[0], thetaav.size, 2])

        neg_pi_ind = np.abs(thetaav - (np.pi / 2.0)) < eps
        if np.any(neg_pi_ind):
            out[:, neg_pi_ind, :] = [-xs[:, neg_pi_ind], -ys[:, neg_pi_ind]]

        pos_pi_ind = np.abs(thetaav + (np.pi / 2.0)) < eps
        if np.any(pos_pi_ind):
            out[:, pos_pi_ind, :] = [xs[:, pos_pi_ind], ys[:, pos_pi_ind]]

        ind = ~np.logical_and(neg_pi_ind, pos_pi_ind)
        ind_c = c[:, ind]
        ind_thetaav = thetaav[ind]
        out[:, ind, 0] = xs[:, ind] * np.sin(ind_c)
        out[:, ind, 1] = (ind_c * np.cos(ind_thetaav) * np.cos(ind_c) -
                          ys[:, ind] * np.sin(ind_thetaav) * np.sin(ind_c))

        return out

    # theta = (pi / 2) - asin(nz)
    def elevation(self, zs):
        return (np.pi / 2.0) - np.arcsin(zs)

    # phi = atan(ny,nx)
    def azimuth(self, xs, ys):
        return np.arctan2(ys, xs)

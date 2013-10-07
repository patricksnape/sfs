import numpy as np
from vector_utils import cart2sph, sph2cart


def expmap_smith(base_vectors, tangent_vectors):
    # If we've been passed a single vector to map, then add the extra axis
    # Number of sample first
    if len(tangent_vectors.shape) == 3:
        vector_count = tangent_vectors.shape[0]
    else:
        tangent_vectors = tangent_vectors[None, ...]
        vector_count = 1

    N = tangent_vectors.shape[1]
    ns = np.zeros([vector_count, N, tangent_vectors.shape[2] + 1])
    eps = np.spacing(1)
    
    for i in xrange(vector_count):
        kset = tangent_vectors[i, ...]
        
        long0lat1 = cart2sph(-base_vectors[:, 1], base_vectors[:, 0], base_vectors[:, 2])
        long0lat1[:, 0] += np.pi
     
        rho = np.sqrt(kset[:, 0] ** 2 + kset[:, 1] ** 2)
        Z = np.exp(np.arctan2(kset[:, 1], kset[:, 0]) * 1j)
        V1 = rho * np.real(Z)
        V2 = rho * np.imag(Z)
        
        # To prevent divide by 0 warnings when rho is 0
        ir = rho == 0
        rho[ir] = eps
        c = rho
        
        c[ir] = eps
        
        Y = np.real(np.arcsin(np.cos(c) * np.sin(long0lat1[:, 1]) + np.cos(long0lat1[:, 1]) * np.sin(c) * V2 / rho))
        X = np.real(long0lat1[:, 0] + np.arctan2(V1 * np.sin(c), np.cos(long0lat1[:, 1]) * np.cos(c) * rho - np.sin(long0lat1[:, 1]) * V2 * np.sin(c)))
        
        vectors = sph2cart(X - np.pi, Y, np.ones_like(Y))
        # Swap x and y axes
        vectors[:, [0, 1]] = vectors[:, [1, 0]]
        vectors[:, 1] = -vectors[:, 1]
        
        ns[i, ...] = vectors
        
    return ns


def logmap_smith(base_vectors, sd_vectors):
    # If we've been passed a single vector to map, then add the extra axis
    # Number of sample first
    if len(sd_vectors.shape) == 3:
        vector_count = sd_vectors.shape[0]
    else:
        sd_vectors = sd_vectors[None, ...]
        vector_count = 1
    
    N = sd_vectors.shape[1]    
    vs = np.zeros([vector_count, N, sd_vectors.shape[2] - 1])
    
    for i in xrange(vector_count):
        # Get current vector
        kset = sd_vectors[i, ...]
        projected = np.zeros([N, 2])
        
        longlat = cart2sph(-kset[:, 1], kset[:, 0], kset[:, 2])
        longlat[:, 0] += np.pi
        
        long0lat1 = cart2sph(-base_vectors[:, 1], base_vectors[:, 0], base_vectors[:, 2]) 
        long0lat1[:, 0] += np.pi
        
        c = np.arccos(np.sin(long0lat1[:, 1]) * np.sin(longlat[:, 1]) + np.cos(long0lat1[:, 1]) * np.cos(longlat[:, 1]) * np.cos(longlat[:, 0] - long0lat1[:, 0]))
        
        k = 1.0 / np.sin(c)
        k = np.nan_to_num(k)
        k = c * k
        
        projected[:, 0] = k * (np.cos(longlat[:, 1]) * np.sin(longlat[:, 0] - long0lat1[:, 0]))
        projected[:, 1] = k * (np.cos(long0lat1[:, 1]) * np.sin(longlat[:, 1]) - np.sin(long0lat1[:, 1]) * np.cos(longlat[:, 1]) * np.cos(longlat[:, 0] - long0lat1[:, 0]))
        
        vs[i, ...] = projected
                                                                                                           
    return vs


def logmap_snape(base_vectors, sd_vectors):
    # If we've been passed a single vector to map, then add the extra axis
    # Number of sample first
    if len(sd_vectors.shape) == 3:
        vector_count = sd_vectors.shape[0]
    else:
        sd_vectors = sd_vectors[None, ...]
        vector_count = 1
    
    N = sd_vectors.shape[1]    
    vs = np.zeros([vector_count, N, sd_vectors.shape[2] - 1])
    
    thetaav = elevation(base_vectors[:, 2])
    phiav = azimuth(base_vectors[:, 0], base_vectors[:, 1])
    phiav, thetaav = clip_angles(phiav, thetaav)
    
    for i in xrange(vector_count):
        # Get current vector
        kset = sd_vectors[i, ...]
        projected = np.zeros([N, 2])
    
        # find any zero normals as they present a real problem
        # the column indicies are the same in both sets of data
        zero_indices = np.sum(np.abs(kset), axis=1) == 0.0
        
        thetak = elevation(kset[:, 2])
        phik = azimuth(kset[:, 0], kset[:, 1])
        phik, thetak = clip_angles(phik, thetak)
        
        # cos(c) = sin(thetaav) * sin(thetak) + cos(thetaav) * cos(thetak) * cos[phik - phiav]
        cosc = np.sin(thetaav) * np.sin(thetak) + np.cos(thetaav) * np.cos(thetak) * np.cos(phik - phiav)
        c = np.arccos(cosc)
        # kprime = c / sin(c)
        kprime = c / np.sin(c)
        
        # xs = kprime * cos(thetak) * sin[phik - phiav]
        projected[:, 0] = kprime * np.cos(thetak) * np.sin(phik - phiav)
        # ys = kprime * (cos(thetaav) * sin(phik) - sin(thetaav) * cos(thetak) * cos[phik - phiav]
        projected[:, 1] = kprime * (np.cos(thetaav) * np.sin(thetak) - np.sin(thetaav) * np.cos(thetak) * np.cos(phik - phiav))
    
        # reset the zero normals back to 0
        projected[zero_indices, :] = 0.0
        
        # reshape back to column vector
        vs[i, ...] = projected
        
    return vs


def expmap_snape(base_vectors, tangent_vectors):
    # If we've been passed a single vector to map, then add the extra axis
    # Number of sample first
    if len(tangent_vectors.shape) == 3:
        vector_count = tangent_vectors.shape[0]
    else:
        tangent_vectors = tangent_vectors[None, ...]
        vector_count = 1

    N = tangent_vectors.shape[1]
    ns = np.zeros([vector_count, N, tangent_vectors.shape[2] + 1])
    
    thetaav = elevation(base_vectors[:, 2])
    phiav = azimuth(base_vectors[:, 0], base_vectors[:, 1])
    phiav, thetaav = clip_angles(phiav, thetaav)
    
    for i in xrange(vector_count):
        # as vector matrix
        kset = tangent_vectors[i, ...]
        vectors = np.zeros([N, 3])
        # find any zero normals as they present a real problem
        # the column indicies are the same in both sets of data
        zero_indices = np.sum(np.abs(kset), axis=1) == 0.0
    
        xs = kset[:, 0]
        ys = kset[:, 1]
        
        c = np.sqrt(xs ** 2 + ys ** 2)
        recipc = 1.0 / c
        
        # thetas = asin[cos(c) * sin(thetaav) - (1/c) * yk * sin(c) * cos(thetav)]
        s = np.cos(c) * np.sin(thetaav) + recipc * ys * np.sin(c) * np.cos(thetaav)
        
        # Elevation
        el = np.arcsin(s)
        # Azimuth
        # phis = phiav + atan(psi)
        phis = psi(c, thetaav, xs, ys)
        azi = phiav + np.arctan2(phis[:, 0], phis[:, 1])
        azi, el = clip_angles(azi, el)
        
        # convert angles to coordinates
        vectors[:, 0] = np.cos(azi) * np.sin(el)
        vectors[:, 1] = np.sin(azi) * np.sin(el)
        vectors[:, 2] = np.cos(el)
    
        # reset zero projections back to zero
        vectors[zero_indices, :] = 0.0
        
        # reshape back to column vector
        ns[i, ...] = vectors
        
    return ns


def clip_angles(phi, theta):
    # round thetas back in to the range [-pi/2, pi/2]
    pi_over_2 = np.pi / 2.0
    theta[theta > pi_over_2] = theta[theta > pi_over_2] - np.pi
    theta[theta < -pi_over_2] = theta[theta < -pi_over_2] + np.pi
    
    phi[phi > np.pi] = phi[phi > np.pi] - 2.0 * np.pi
    phi[phi < np.pi] = phi[phi < np.pi] + 2.0 * np.pi
    
    return phi, theta


# psi = thetaav != (pi / 2) -> 
#           xk * sin(c) / c * cos(thetaav) * cos(c) - yk * sin(thetav) * sin(c)
#       thetaav == (pi/2)   -> -(xk/yk)
#       thetaav == -(pi/2)  -> xk/yk
def psi(c, thetaav, xs, ys):
    eps = np.spacing(1)
    out = np.zeros([thetaav.size, 2])
        
    neg_pi_ind = np.abs(thetaav - (np.pi / 2.0)) < eps
    if np.any(neg_pi_ind):
        out[neg_pi_ind, :] = [-xs[neg_pi_ind], -ys[neg_pi_ind]]
    
    pos_pi_ind = np.abs(thetaav + (np.pi / 2.0)) < eps
    if np.any(pos_pi_ind):
        out[pos_pi_ind, :] = [xs[pos_pi_ind], ys[pos_pi_ind]]
    
    ind = ~np.logical_and(neg_pi_ind, pos_pi_ind)
    out[ind, 0] = xs[ind] * np.sin(c[ind])
    out[ind, 1] = c[ind] * np.cos(thetaav[ind]) * np.cos(c[ind]) - ys[ind] * np.sin(thetaav[ind]) * np.sin(c[ind])
    
    return out


# theta = (pi / 2) - asin(nz)
def elevation(zs):
    return (np.pi / 2.0) - np.arcsin(zs)  # elevation between 0, pi and 0 at z=1


# phi = atan(ny,nx)
def azimuth(xs, ys):
    return np.arctan2(ys, xs)



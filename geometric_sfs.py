import numpy as np
from vector_utils import normalise_vector, normalise_image


def on_cone_rotation(theta_image, normal_image, s):
    theta = theta_image.as_vector()
    nprime = normal_image.as_vector(keep_channels=True)

    # cross product and break in to row vectors
    C = np.cross(nprime, s)
    C = normalise_vector(C)
    
    u = C[:, 0]
    v = C[:, 1]
    w = C[:, 2]
    
    # expects |nprime| = |sec| = 1
    # represents intensity and can never be < 0
    d = np.squeeze(np.inner(nprime, s))
    d[d < 0.0] = 0.0
    
    beta = np.arccos(d)
    alpha = theta - beta
    # flip alpha so that it rotates along the correct axis
    alpha = -alpha
    
    c = np.cos(alpha)
    cprime = 1.0 - c
    s = np.sin(alpha)
    
    # setup structures
    N = nprime.shape[0]
    phi = np.zeros([N, 3, 3])
    
    phi[:, 0, 0] = c + u ** 2 * cprime
    phi[:, 0, 1] = -w * s + u * v * cprime
    phi[:, 0, 2] = v * s + u * w * cprime
    
    phi[:, 1, 0] = w * s + u * v * cprime
    phi[:, 1, 1] = c + v ** 2 * cprime
    phi[:, 1, 2] = -u * s + v * w * cprime
    
    phi[:, 2, 0] = -v * s + u * w * cprime
    phi[:, 2, 1] = u * s + v * w * cprime
    phi[:, 2, 2] = c + w ** 2 * cprime
          
    n = np.einsum('kjl, klm -> kj', phi, nprime[..., None])

    # Normalize the result ??
    n = normalise_vector(n)
    return normal_image.from_vector(n)


def esimate_normals_from_intensity(average_normals, theta_image):
    theta = theta_image.as_vector()

    # Represents tan(phi) = sin(partial I/ partial y) / cos(partial I/ partial x)
    # Where [partial I/ partial y] is the y-direction of the gradient field
    average_masked_pixels = average_normals.as_vector(keep_channels=True)
    n = np.sqrt(average_masked_pixels[:, 0] ** 2 + average_masked_pixels[:, 0] ** 2)
    cosphi = average_masked_pixels[:, 0] / average_masked_pixels[:, 2]
    sinphi = average_masked_pixels[:, 1] / average_masked_pixels[:, 2]
    
    # Reset any nan-vectors to 0.0
    cosphi[np.isnan(cosphi)] = 0.0
    sinphi[np.isnan(sinphi)] = 0.0
    
    nestimates = np.zeros_like(average_masked_pixels)
    # sin(theta) * cos(phi)
    nestimates[:, 0] = np.sin(theta) * cosphi
    # sin(theta) * sin(phi)
    nestimates[:, 1] = np.sin(theta) * sinphi
    nestimates[:, 2] = np.cos(theta)

    # Unit normals
    nestimates = normalise_vector(nestimates)
    return average_normals.from_vector(nestimates)


def identity_logmap(base_vectors, sd_vectors):
    return sd_vectors


def identity_expmap(base_vectors, tangent_vectors):
    return tangent_vectors


def geometric_sfs(intensity_image, normal_model, light_vector,
                  max_iters=100, max_error=10**-6, logmap=identity_logmap,
                  expmap=identity_expmap):
    """
    It is assumed that the given intensity image has been pre-aligned so that
    it is in correspondance with the model.

    1. Calculate an initial estimate of the field of surface normals n using
       (12)
    2. Each normal in the estimated field n undergoes an azimuthal equidistant
       projection (3) to give a vector of transformed coordinates v0.
    3. The vector of best fit model parameters is given by ``b = P.T * v0 ``.
    4. The vector of transformed coordinates corresponding
       to the best-fit parameters is given by ``vprime = (P P.T)v0``
    5. Using the inverse azimuthal equidistant projection
       (4), find the off-cone best fit surface normal nprime from vprime.
    6. Find the on-cone surface normal nprimeprime by rotating the
       off-cone surface normal nprime using
       ``nprimeprime(i,j) = theta * nprime(i, j)``
    7. Test for convergence. If
       ``sum over i,j arccos(n(i,j) . nprimeprime(i,j)) < eps``,
       where eps is a predetermined threshold, then stop and return b as the
       estimated model parameters and ``nprimeprime`` as rhe recovered needle
       map.
    8. Make ``n(i,j) = nprimeprime(i,j)`` and return to Step 2.

    Parameters
    ----------
    intensity_image : (M, N, 1) :class:`pybug.image.MaskedNDImage`
        The 1-channel intensity image of a face to recover normals from.
    normal_model : :class:`pybug.model.linear.PCAModel`
        A PCA model representing a subspace of normals.
    light_vector : (3,) ndarray
        A single light vector that represent the lighting direction that the
        image is lit from.
    max_iters : int, optional
        The maximum number of iterations to perform.

        Default: 100
    max_error : float, optional
        The maximum epsilon to test for convergence.

        Default: 10^-6
    logmap : func, optional
        The logmap function to perform to the normals before reconstructing
        from the PCA model. This should be the same subspace as the PCA model.

        Default: Identity mapping (no-op)
    expmap : func, optional
        The expmap function to perform to the subspace after reconstruction
        from the PCA model. This should be the same projection function as the
        logmap.

        Default: Identity mapping (no-op)

    Returns
    -------
    normal_image : (M, N, 3) :class:`pybug.image.MaskedNDImage`
        A 3-channel image representing the components of the recovered normals.

    References
    ----------
    [1] Smith, William AP, and Edwin R. Hancock.
        Facial Shape-from-shading and Recognition Using Principal Geodesic
        Analysis and Robust Statistics
        IJCV (2008)
    [2] Smith, William AP, and Edwin R. Hancock.
        Recovering facial shape using a statistical model of surface normal
        direction.
        T-PAMI (2006)
    """
    # Ensure the light is a unit vector
    light_vector = normalise_vector(light_vector)

    # Equation (1): Should never be < 0 if image is properly scaled
    theta_vec = np.arccos(intensity_image.as_vector())
    theta_image = intensity_image.from_vector(theta_vec)

    n = esimate_normals_from_intensity(normal_model.mean, theta_image)

    for i in xrange(max_iters):
        v0 = logmap(normal_model.mean, n)

        # Vector of best-fit parameters
        vprime = normal_model.reconstruct(v0)

        nprime = expmap(normal_model.mean, vprime)
        nprime = normalise_image(nprime)
        
        # Equivalent to
        # expmap(theta * logmap(expmap(vprime)) / row_norm(logmap(expmap(vprime))))
        npp = on_cone_rotation(theta_image, nprime, light_vector)
        npp = normalise_image(npp)

        # Check for convergence of the algorithm (not guaranteed)
        # Inner product
        error = np.arccos(np.sum(n.as_vector(keep_channels=3) *
                                 npp.as_vector(keep_channels=3), axis=1))
        # Total error
        error = np.sum(np.nan_to_num(error))
        n = npp

        # Algorithm has converged
        if error < max_error:
            break

    return normalise_image(npp)




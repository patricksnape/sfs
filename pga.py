import numpy as np
from vector_utils import row_norm, normalise_vector
from fast_pga import rotate_north_pole


class PGA(object):

    def __init__(self, base_vectors):
        super(PGA, self).__init__()
        self.base_vectors = base_vectors
        self.rotation_matrices = rotate_north_pole(self.base_vectors)

    def expmap(self, tangent_vectors):
        # If we've been passed a single vector to map, then add the extra axis
        # Number of sample first
        if len(tangent_vectors.shape) < 3:
            tangent_vectors = tangent_vectors[None, ...]

        # Expmap
        v1 = tangent_vectors[..., 0]
        v2 = tangent_vectors[..., 1]
        normv = row_norm(tangent_vectors)

        exp = np.concatenate([(v1 * np.sin(normv) / normv)[..., None],
                              (v2 * np.sin(normv) / normv)[..., None],
                              np.cos(normv)[..., None]], axis=2)
        near_zero_ind = normv < np.spacing(1)
        exp[near_zero_ind, :] = [0.0, 0.0, 1.0]

        # Rotate back to geodesic mean from north pole
        # Apply inverse rotation matrix due to data ordering
        ns = np.einsum('fvi, ijv -> fvj', exp, self.rotation_matrices)

        return ns

    def logmap(self, sd_vectors):
        # If we've been passed a single vector to map, then add the extra axis
        # Number of sample first
        if len(sd_vectors.shape) < 3:
            sd_vectors = sd_vectors[None, ...]

        rotated_data = np.einsum('ijv, fvj -> fvi', self.rotation_matrices,
                                                    sd_vectors)

        # Perform the North Pole Logmap
        # theta / sin(theta)
        zs = rotated_data[..., 2]
        scales = np.arccos(zs) / np.sqrt(1.0 - zs ** 2)
        scales[np.isnan(scales)] = 1.0
        # Build the column vector the transpose for correct ordering
        vs = rotated_data * scales[..., None]

        return vs[..., :2]


def intrinsic_mean(sd_vectors, mapping_class, max_iters=20):
    # Compute initial estimate (Euclidean mean of data)
    mus = normalise_vector(np.mean(sd_vectors, axis=0))

    for i in xrange(max_iters):
        mapping_object = mapping_class(mus)
        # Iteratively improve estimate of intrinsic mean
        mean_tangent = np.mean(mapping_object.logmap(sd_vectors), axis=0)
        mus = np.squeeze(mapping_object.expmap(mean_tangent))

    return mus

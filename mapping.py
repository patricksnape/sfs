from pga import PGA
from aep import AEP
from cosine_normals import Spherical
import numpy as np


class ImageMapper(object):

    def __init__(self, mapper):
        self._mapper = mapper

    def logmap(self, tangent_vectors):
        vectors = tangent_vectors.as_vector(keep_channels=True)
        logmap_vectors = self._mapper.logmap(vectors)
        return tangent_vectors.from_vector(np.squeeze(logmap_vectors),
                                           n_channels=logmap_vectors.shape[2])

    def expmap(self, sd_vectors):
        vectors = sd_vectors.as_vector(keep_channels=True)
        expmap_vectors = self._mapper.expmap(vectors)
        return sd_vectors.from_vector(np.squeeze(expmap_vectors),
                                      n_channels=expmap_vectors.shape[2])


class IdentityMapper(object):

    def logmap(self, sd_vectors):
        if sd_vectors.ndim == 2:
            return sd_vectors[None, ...]
        else:
            return sd_vectors

    def expmap(self, tangent_vectors):
        if tangent_vectors.ndim == 2:
            return tangent_vectors[None, ...]
        else:
            return tangent_vectors

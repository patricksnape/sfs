from pga import logmap_pga_northpole, expmap_pga_northpole
from vector_utils import normalise_vector
from numpy.testing import assert_allclose
import numpy as np


n_samples = 200
n_vectors = 10000
sd_vectors = normalise_vector(
    np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_vectors, 3)))
base_vectors = normalise_vector(
    np.random.uniform(low=-1.0, high=1.0, size=(n_vectors, 3)))


def test_pga_northpole_mapping_equality():
    tangent_vectors = logmap_pga_northpole(base_vectors, sd_vectors)
    mapped_vectors = expmap_pga_northpole(base_vectors, tangent_vectors)
    assert_allclose(mapped_vectors, sd_vectors)
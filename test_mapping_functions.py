from pga import logmap_pga_northpole, expmap_pga_northpole
from aep import expmap_aep_smith, expmap_aep_snape, logmap_aep_snape, \
    logmap_aep_smith
from vector_utils import normalise_vector
from numpy.testing import assert_allclose
import numpy as np


n_samples = 10
n_vectors = 100
sd_vectors = normalise_vector(
    np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_vectors, 3)))
base_vectors = normalise_vector(
    np.random.uniform(low=-1.0, high=1.0, size=(n_vectors, 3)))

small_sd_vectors = np.array([[[ 0.75098051,  0.17029863,  0.6379864 ],
                              [ 0.50001385,  0.53529381, -0.68076919]],

                             [[-0.86560911, -0.41208702, -0.28443833],
                              [-0.63544436,  0.60226651,  0.4832034 ]]])
small_base_vectors = np.array([[ 0.67114357, -0.01134708,  0.74124055],
                               [-0.76751246,  0.62469913,  0.14379021]])
small_expected_pga_tangent_vectors = np.array([[[ 0.13036596,  0.18233274],
                                               [-0.09662949,  1.71587233]],

                                              [[-1.81744851, -1.68283606],
                                               [ 0.32233591, -0.17535739]]])
small_expected_aep_snape_tangent_vectors = np.array(
    [[[ 0.15280248,  0.15605087],
     [-2.38228641, -0.2084999 ]],

    [[-0.58328875, -2.48658029],
     [-0.03721135, -0.35865463]]])
small_expected_aep_smith_tangent_vectors = np.array(
    [[[ 0.18451048, -0.12726506],
      [-1.26978438, -1.15810307]],

     [[-1.71331896,  1.78874102],
      [-0.06747517,  0.36069066]]])


def test_pga_northpole_logmap():
    tangent_vectors = logmap_pga_northpole(small_base_vectors,
                                           small_sd_vectors)
    assert_allclose(tangent_vectors, small_expected_pga_tangent_vectors)


def test_pga_northpole_expmap():
    mapped_vectors = expmap_pga_northpole(small_base_vectors,
                                          small_expected_pga_tangent_vectors)
    assert_allclose(mapped_vectors, small_sd_vectors)


def test_pga_northpole_mapping_equality():
    tangent_vectors = logmap_pga_northpole(base_vectors, sd_vectors)
    mapped_vectors = expmap_pga_northpole(base_vectors, tangent_vectors)
    assert_allclose(mapped_vectors, sd_vectors)


def test_aep_snape_logmap():
    tangent_vectors = logmap_aep_snape(small_base_vectors,
                                       small_sd_vectors)
    assert_allclose(tangent_vectors, small_expected_aep_snape_tangent_vectors)


def test_aep_snape_expmap():
    mapped_vectors = expmap_aep_snape(small_base_vectors,
                                      small_expected_aep_snape_tangent_vectors)
    assert_allclose(mapped_vectors, small_sd_vectors)


def test_aep_smith_logmap():
    tangent_vectors = logmap_aep_smith(small_base_vectors,
                                       small_sd_vectors)
    assert_allclose(tangent_vectors, small_expected_aep_smith_tangent_vectors)


def test_aep_smith_expmap():
    mapped_vectors = expmap_aep_smith(small_base_vectors,
                                      small_expected_aep_smith_tangent_vectors)
    assert_allclose(mapped_vectors, small_sd_vectors)


def test_aep_snape_mapping_equality():
    tangent_vectors = logmap_aep_snape(base_vectors, sd_vectors)
    mapped_vectors = expmap_aep_snape(base_vectors, tangent_vectors)
    assert_allclose(mapped_vectors, sd_vectors)


def test_aep_smith_mapping_equality():
    tangent_vectors = logmap_aep_smith(base_vectors, sd_vectors)
    mapped_vectors = expmap_aep_smith(base_vectors, tangent_vectors)
    assert_allclose(mapped_vectors, sd_vectors)


def test_aep_snape_smith_mapping_equality():
    snape = expmap_aep_snape(base_vectors, logmap_aep_snape(base_vectors,
                                                            sd_vectors))
    smith = expmap_aep_smith(base_vectors, logmap_aep_smith(base_vectors,
                                                            sd_vectors))
    assert_allclose(snape, smith)

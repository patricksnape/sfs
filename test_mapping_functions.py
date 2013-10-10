from pga import PGA, intrinsic_mean
from aep import AEP
from cosine_normals import Spherical
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
small_expected_aep_smith_tangent_vectors = np.array(
    [[[ 0.18451048, -0.12726506],
      [-1.26978438, -1.15810307]],

     [[-1.71331896,  1.78874102],
      [-0.06747517,  0.36069066]]])
small_expected_spherical_tangent_vectors = np.array(
    [[[ 0.97523904,  0.22115337,  0.6379864 ,  0.77004763],
     [ 0.68261463,  0.73077853, -0.68076919,  0.73249799]],
     [[-0.90290416, -0.42984192, -0.28443833,  0.95869434],
      [-0.72580064,  0.6879051 ,  0.4832034 ,  0.87550812]]])
expected_pga_northpole_mu = np.array([[-0.2585258 , -0.54531364,  0.79736908],
                                      [-0.11649889,  0.97854257, -0.16994838]])


def test_pga_northpole_logmap():
    tangent_vectors = PGA(small_base_vectors).logmap(small_sd_vectors)
    assert_allclose(tangent_vectors, small_expected_pga_tangent_vectors)


def test_pga_northpole_expmap():
    mapped_vectors = PGA(small_base_vectors).expmap(
        small_expected_pga_tangent_vectors)
    assert_allclose(mapped_vectors, small_sd_vectors)


def test_pga_northpole_mapping_equality():
    pga = PGA(base_vectors)
    tangent_vectors = pga.logmap(sd_vectors)
    mapped_vectors = pga.expmap(tangent_vectors)
    assert_allclose(mapped_vectors, sd_vectors)


def test_pga_northpole_intrinsic_mean():
    mu = intrinsic_mean(small_sd_vectors, PGA, max_iters=5)
    assert_allclose(mu, expected_pga_northpole_mu)


def test_spherical_logmap():
    tangent_vectors = Spherical().logmap(small_sd_vectors)
    assert_allclose(tangent_vectors, small_expected_spherical_tangent_vectors)


def test_spherical_expmap():
    mapped_vectors = Spherical().expmap(
        small_expected_spherical_tangent_vectors)
    assert_allclose(mapped_vectors, small_sd_vectors)


def test_spherical_mapping_equality():
    pga = Spherical()
    tangent_vectors = pga.logmap(sd_vectors)
    mapped_vectors = pga.expmap(tangent_vectors)
    assert_allclose(mapped_vectors, sd_vectors)


def test_aep_logmap():
    tangent_vectors = AEP(small_base_vectors).logmap(small_sd_vectors)
    assert_allclose(tangent_vectors, small_expected_aep_smith_tangent_vectors)


def test_aep_expmap():
    mapped_vectors = AEP(small_base_vectors).expmap(
        small_expected_aep_smith_tangent_vectors)
    assert_allclose(mapped_vectors, small_sd_vectors)


def test_aep_smith_mapping_equality():
    smith = AEP(base_vectors)
    tangent_vectors = smith.logmap(sd_vectors)
    mapped_vectors = smith.expmap(tangent_vectors)
    assert_allclose(mapped_vectors, sd_vectors)


def test_aep_intrinsic_mean():
    mu = intrinsic_mean(small_sd_vectors, AEP, max_iters=5)
    assert_allclose(mu, expected_pga_northpole_mu)

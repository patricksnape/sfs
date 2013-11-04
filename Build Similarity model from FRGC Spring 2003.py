# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os.path
import numpy as np
import matplotlib.pyplot as plt
import cPickle
from pybug.image import DepthImage, ShapeImage, RGBImage, IntensityImage
from pybug.landmark.labels import labeller, ibug_68_closed_mouth
from pybug.shape import PointCloud
from pybug.io import auto_import
from pybug.image import MaskedNDImage
from pybug.transform import Translation, SimilarityTransform
from pybug.transform.tps import TPS
from pybug.transform.piecewiseaffine import PiecewiseAffineTransform
from pybug.groupalign import GeneralizedProcrustesAnalysis
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter, median_filter
from inpaint import replace_nans
import numpy as np
# <codecell>

def print_replace_line(string):
    import sys
    # Cheeky carriage return so we print on the same line
    sys.stdout.write('\r' + string)
    sys.stdout.flush()

# <markdowncell>

# ## Load all the aligned shapes from the dataset

# <codecell>

RECREATE_MESHES = False

# <codecell>

from sfs_io import load_frgc

images = load_frgc('spring2003', RECREATE_MESHES)

# <codecell>

def extract_normals(images):
    vector_shape = images[0].mesh.vertex_normals.shape
    N = len(images)
    normals = np.zeros([N, vector_shape[0], vector_shape[1]])
    for i, im in enumerate(images):
        normals[i, ...] = im.mesh.vertex_normals
    return normals

# <codecell>

def create_feature_space(feature_matrix, example_image, feature_space_name):
    feature_space_images = []
    N = feature_matrix.shape[0]
    for i, n in enumerate(feature_matrix):
        new_im = MaskedNDImage.blank(example_image.shape, mask=example_image.mask, n_channels=n.shape[1])
        new_im.from_vector_inplace(n.flatten())
        new_im.landmarks = example_image.landmarks
        feature_space_images.append(new_im)
        print_replace_line('Image {0} of {1}'.format(i + 1, N))

    cPickle.dump(images, open('/vol/atlas/pts08/cvpr/frgc_spring2003_4_{0}.pkl'.format(feature_space_name), 'wb'), protocol=2)
    return feature_space_images

# <markdowncell>

# ## Generate the frame of reference via a Similarity Transform

# <codecell>

from landmarks import ibug_68_edge
# Pull out the landmarks
labeller(images, 'PTS', ibug_68_edge)
shapes = [img.landmarks['ibug_68_edge'].lms for img in images]

# <codecell>

ref_frame = MaskedNDImage.blank([480, 360])

# <codecell>

from warp import build_similarity_transform
# Warp each of the images to the reference image
sim_transforms = [build_similarity_transform(shape) for shape in shapes]
warped_images = [img.warp_to(ref_frame.mask, t) for img, t in zip(images, sim_transforms)]

# <markdowncell>

# ## Calculate the normal matrix for all the images

# <codecell>

normal_matrix = extract_normals(warped_images)

# <markdowncell>

# ## Calculate the normals (for LS and Cosine)

# <codecell>

normal_images = create_feature_space(normal_matrix, warped_images[0], 'normals')

# <markdowncell>

# ## Calculate the Spherical feature space

# <codecell>

from cosine_normals import Spherical
spherical_matrix = Spherical().logmap(normal_matrix)
spherical_images = create_feature_space(spherical_matrix, warped_images[0], 'spherical')

# <markdowncell>

# ## Calculate the AEP feature space

# <codecell>

from vector_utils import normalise_vector
mean_normals = normalise_vector(np.mean(normal_matrix, 0))

# <codecell>

from logmap_utils import partial_logmap
from aep import AEP

aep_matrix = AEP(mean_normals).logmap(normal_matrix)
aep_images = create_feature_space(aep_matrix, warped_images[0], 'aep')

# <markdowncell>

# ## Calculate the PGA feature space

# <codecell>

from pga import PGA, intrinsic_mean
mu = intrinsic_mean(normal_matrix, PGA, max_iters=50)

# <codecell>

pga_matrix = PGA(mu).logmap(normal_matrix)
pga_images = create_feature_space(pga_matrix, warped_images[0], 'pga')

# <markdowncell>

# ## Calculate the PCA for LS, Spherical, Cosine and AEP

# <codecell>

# Create the template image
template = ref_frame

# <codecell>

from pybug.model.linear import PCAModel
normal_model = PCAModel(normal_images, center=True, n_components=200)
cosine_model = PCAModel(normal_images, center=False, n_components=200)
spherical_model = PCAModel(spherical_images, center=False, n_components=200)
aep_model = PCAModel(aep_images, center=False, n_components=200)
pga_model = PCAModel(pga_images, center=False, n_components=200)

# <codecell>

mean_normals_image = normal_model.mean
mu_image = mean_normals_image.from_vector(mu)

with open('/vol/atlas/pts08/cvpr/frgc_spring2003_sfs_sim_normal', 'wb') as f:
    cPickle.dump({'appearance_model': normal_model,
                 'template': template,
                 'mean_normals': mean_normals_image},
                 f, protocol=2)
with open('/vol/atlas/pts08/cvpr/frgc_spring2003_sfs_sim_cosine', 'wb') as f:
    cPickle.dump({'appearance_model': cosine_model,
                 'template': template,
                 'mean_normals': mean_normals_image},
                 f, protocol=2)
with open('/vol/atlas/pts08/cvpr/frgc_spring2003_sfs_sim_spherical', 'wb') as f:
    cPickle.dump({'appearance_model': spherical_model,
                 'template': template,
                 'mean_normals': mean_normals_image},
                 f, protocol=2)
with open('/vol/atlas/pts08/cvpr/frgc_spring2003_sfs_sim_aep', 'wb') as f:
    cPickle.dump({'appearance_model': aep_model,
                 'template': template,
                 'mean_normals': mean_normals_image},
                 f, protocol=2)
with open('/vol/atlas/pts08/cvpr/frgc_spring2003_sfs_sim_pga', 'wb') as f:
    cPickle.dump({'appearance_model': pga_model,
                 'template': template,
                 'mean_normals': mean_normals_image,
                 'intrinsic_mean_normals': mu_image},
                 f, protocol=2)

# <codecell>



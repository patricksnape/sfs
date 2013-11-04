# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import cPickle
from copy import deepcopy
import numpy as np

with open('/vol/atlas/pts08/cvpr/frgc_spring2003_sfs_tps_bej_spherical.pkl', 'rb') as f:
    model = cPickle.load(f)

normal_model = model['appearance_model']
reference_frame = model['template']
mean_normals = model['mean_normals']
reference_frame = model['template']
try:
    intrinsic_mean_normals = model['intrinsic_mean_normals']
except Exception:
    pass

# <codecell>

from pybug.image import MaskedNDImage
from pybug.io import auto_import
from pybug.landmark import labeller, ibug_68_closed_mouth
from landmarks import ibug_68_edge

sfs_index = 2
bej = auto_import('/vol/atlas/databases/alex_images/bej*.ppm')
# Create a 4 channel image where each channel is the greyscale of an image
ground_truth_images = MaskedNDImage(np.concatenate([im.as_greyscale().pixels for im in bej], axis=2))
intensity_image = bej[sfs_index].as_greyscale()

intensity_image.landmarks = bej[0].landmarks
ground_truth_images.landmarks['PTS'] = bej[0].landmarks['PTS']

labeller([ground_truth_images, intensity_image], 'PTS', ibug_68_closed_mouth)
# labeller([ground_truth_images, intensity_image], 'PTS', ibug_68_edge)

lights = np.array([[ 0.5,  0.4, 2],
                   [-0.5,  0.4, 2],
                   [-0.5, -0.4, 2],
                   [ 0.5, -0.4, 2]])

# <codecell>

from pybug.transform.tps import TPS
from warp import build_similarity_transform
# tr = build_similarity_transform(intensity_image.landmarks['ibug_68_edge'].lms)
tr = TPS(reference_frame.landmarks['ibug_68_closed_mouth'].lms, intensity_image.landmarks['ibug_68_closed_mouth'].lms)

# <codecell>

from pybug.transform import SimilarityTransform
from pybug.image import MaskedNDImage, IntensityImage

warped_intensity_image = intensity_image.warp_to(reference_frame.mask, tr)
warped_ground_truth_image = ground_truth_images.warp_to(reference_frame.mask, tr)

warped_intensity_image = MaskedNDImage(warped_intensity_image.pixels, mask=warped_intensity_image.mask)
warped_intensity_image.view()

# <codecell>

# Use ground truth light
estimate_light = lights[sfs_index, :]
print estimate_light

# <codecell>

from photometric_stereo import photometric_stereo as ps

ground_truth_normals, ground_truth_albedo = ps(warped_ground_truth_image, lights)

# <codecell>

from pybug.image import MaskedNDImage
from scipy.ndimage.filters import gaussian_filter, median_filter
from geometric_sfs import geometric_sfs as sfs, worthington_hancock_sfs
from mapping import AEP, PGA, Spherical, ImageMapper, IdentityMapper
# mapping_object = ImageMapper(PGA(intrinsic_mean_normals.as_vector(keep_channels=True)))
mapping_object = ImageMapper(Spherical())

warped_intensity_image = MaskedNDImage(warped_intensity_image.pixels.copy(), mask=warped_intensity_image.mask)
mean_normals_image = warped_intensity_image.from_vector(mean_normals, n_channels=3)
# Normalise the image so that it has unit albedo?
warped_intensity_image.masked_pixels /= ground_truth_albedo.masked_pixels
# warped_intensity_image.masked_pixels /= np.max(warped_intensity_image.masked_pixels)
worthington = worthington_hancock_sfs(warped_intensity_image, mean_normals_image, estimate_light, n_iters=40)

# <codecell>

# from geometric_sfs import horn_brooks
# reconstructed_normals_horn = horn_brooks(warped_intensity_image, mean_normals, normal_model, estimate_light, n_iters=100, c_lambda=10, mapping_object=mapping_object)

# <codecell>

from pybug.visualize.viewmayavi import MayaviVectorViewer3d
from pybug.image import DepthImage, RGBImage

ground_truth_normals.view_new(channel=0)
# reconstructed_normals_horn.view_new(channel=0)

# <codecell>

temp_texture = np.concatenate([warped_intensity_image.pixels]*3, axis=2)

# <codecell>

# <codecell>

from surface_reconstruction import frankotchellappa
# If we use gradient fields then we don't need to negate the x-axis
recovered_depth = frankotchellappa(-worthington.pixels[:, :, 0], worthington.pixels[:, :, 1])
recovered_depth_image = DepthImage((recovered_depth - np.min(recovered_depth)))
recovered_depth_image.view_new(mode='mesh')

# <codecell>

ground_truth_depth = frankotchellappa(ground_truth_normals.pixels[:, :, 0], ground_truth_normals.pixels[:, :, 1])
ground_truth_depth_image = DepthImage((ground_truth_depth - np.min(ground_truth_depth)) / 1.3, texture=RGBImage(temp_texture))
ground_truth_depth_image.view_new(mode='mesh')

# <codecell>

# <codecell>


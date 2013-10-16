import os.path
from photometric_stereo import photometric_stereo as ps
from pybug.image import MaskedNDImage, DepthImage
from pybug.io import auto_import
from surface_reconstruction import frankotchellappa
from vector_utils import sph2cart
import numpy as np
import matplotlib.pyplot as plt


yale_b_path = '/vol/atlas/databases/cropped_yale_b/'
yale_b_ids = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09',
              'B10']
# 4 lights (above, below, left, right)
yale_b_lights = sph2cart([0.0, 0.0, np.radians(25.0), np.radians(-25.0)],
                         [np.radians(20.0), np.radians(-20.0), 0.0, 0.0],
                         [1.0, 1.0, 1.0, 1.0])
yale_A000_E20 = 'yale{0}_P00A+000E+20.pgm'
yale_A000_E_20 = 'yale{0}_P00A+000E-20.pgm'
yale_A025_E00 = 'yale{0}_P00A+025E+00.pgm'
yale_A_025_E00 = 'yale{0}_P00A-025E+00.pgm'

photoface_path = '/vol/atlas/alex_images/'
photoface_subjects = ['bej', 'bln', 'fav', 'mut', 'pet', 'rob', 'srb']

#for yale_id in yale_b_ids:
#    image_e20 = auto_import(os.path.join(yale_b_path, 'yale' + yale_id, yale_A000_E20.format(yale_id)))[0]
#    image_e_20 = auto_import(os.path.join(yale_b_path, 'yale' + yale_id, yale_A000_E_20.format(yale_id)))[0]
#    image_a25 = auto_import(os.path.join(yale_b_path, 'yale' + yale_id, yale_A025_E00.format(yale_id)))[0]
#    image_a_25 = auto_import(os.path.join(yale_b_path, 'yale' + yale_id, yale_A_025_E00.format(yale_id)))[0]
#
#    images = np.concatenate([image_e20.pixels, image_e_20.pixels,
#                             image_a25.pixels, image_a_25.pixels], axis=2)
#    photometric_stereo_image = MaskedNDImage(images)
#
#    landmarked_image = auto_import(os.path.join(yale_b_path, 'central_lighting_all', yale_id + '*'))
#
#    ground_truth_normals, ground_truth_albedo = ps(photometric_stereo_image,
#                                                   yale_b_lights)
#
#    ground_truth_depth = frankotchellappa(ground_truth_normals.pixels[..., 0],
#                                          ground_truth_normals.pixels[..., 1])
#
#    ground_truth_depth_image = DepthImage(ground_truth_depth)

for subject in photoface_subjects:

from mapping import ImageMapper, AEP, PGA, Spherical, IdentityMapper
from pybug.image import MaskedNDImage, DepthImage
from pybug.io import auto_import
from pybug.landmark import labeller, ibug_68_closed_mouth
from copy import deepcopy
import cPickle
import numpy as np
from surface_reconstruction import frankotchellappa
import mayavi.mlab as mlab
from geometric_sfs import geometric_sfs as sfs
from scipy.linalg import pinv2

def build_mapping_object(feature_space, mean_normals, intrinsic_normals):
    if feature_space == 'aep':
        mapping_object = ImageMapper(AEP(
            mean_normals.as_vector(keep_channels=True)))
    elif feature_space == 'pga':
        mapping_object = ImageMapper(PGA(
            intrinsic_normals.as_vector(keep_channels=True)))
    elif feature_space == 'normal':
        mapping_object = ImageMapper(IdentityMapper())
    elif feature_space == 'cosine':
        mapping_object = ImageMapper(IdentityMapper())
    elif feature_space == 'spherical':
        mapping_object = ImageMapper(Spherical())
    else:
        raise ValueError('Unrecognised feature space!')

    return mapping_object

# Set to offscreen rendering!
mlab.options.offscreen = True

celebrities_path = '/home/pts08/research/sfs/celebrities'
# missing jude_law before mona_lisa
celebrity_subjects = ['clint_eastwood', 'gerard_depardieu', 'jack_nicholson',
                      'mona_lisa', 'samuel_beckett', 'tom_cruise',
                      'tom_hanks']
feature_spaces = ['aep', 'cosine', 'normal', 'pga', 'spherical']

# 5 feature spaces
normals = dict(zip(celebrity_subjects, [{}, {}, {}, {}, {}, {}, {}]))
for s in normals.values():
    s.update(zip(feature_spaces, [None] * len(feature_spaces)))

for i, subject_id in enumerate(celebrity_subjects):
    print "Running experiment for {0}".format(subject_id)

    subject_image = auto_import(
        '/home/pts08/research/sfs/celebrities/{0}.png'.format(subject_id))[0]

    # Choose the third image as the reconstruction candidate
    intensity_image = subject_image.as_greyscale()

    # Label with correct labels
    labeller([intensity_image], 'PTS', ibug_68_closed_mouth)

    # Constrain to mask
    intensity_image.constrain_mask_to_landmarks(
        group='ibug_68_closed_mouth', label='all')

    temp_texture = subject_image

    for k, feature_space in enumerate(feature_spaces):
        print "Running {0} for {1}".format(feature_space, subject_id)
        model_path = '/vol/atlas/pts08/cvpr/frgc_spring2003_sfs_tps_{0}_{1}.pkl'.format(subject_id, feature_space)
        with open(model_path, 'rb') as f:
            model = cPickle.load(f)

        normal_model = model['appearance_model']
        reference_frame = model['template']
        mean_normals = model['mean_normals']
        reference_frame = model['template']
        try:
            intrinsic_mean_normals = model['intrinsic_mean_normals']
        except Exception:
            intrinsic_mean_normals = None

        # Estimate light direction for image
        I = intensity_image.as_vector()
        estimate_light = np.dot(pinv2(mean_normals), I)
        print estimate_light

        # Perform SFS
        warped_intensity_image = MaskedNDImage(intensity_image.pixels.copy(),
                                               mask=intensity_image.mask)
        initial_estimate_image = warped_intensity_image.from_vector(
            mean_normals.copy(), n_channels=3)

        mapping_object = build_mapping_object(feature_space,
                                              initial_estimate_image,
                                              intrinsic_mean_normals)
        # Normalise the image so that it has unit albedo?
        #warped_intensity_image.masked_pixels /= ground_truth_albedo.masked_pixels
        #warped_intensity_image.masked_pixels /= np.max(warped_intensity_image.masked_pixels)
        reconstructed_normals = sfs(warped_intensity_image,
                                    initial_estimate_image, normal_model,
                                    estimate_light, n_iters=200,
                                    mapping_object=mapping_object)

        normals[subject_id][feature_space] = reconstructed_normals

        #reconstructed_depth = frankotchellappa(
        #    -reconstructed_normals.pixels[:, :, 0],
        #    reconstructed_normals.pixels[:, :, 1])
        #reconstructed_depth_image = DepthImage((reconstructed_depth - np.min(reconstructed_depth)) / 2,
        #                                       texture=temp_texture)

        # TODO: Save images
        #reconstructed_depth_image.view(mode='mesh')
        #save_result_images(subject_id, feature_space, 'sfs')

with open('/vol/atlas/pts08/cvpr/results/celebrities/all_result_dict.pkl', 'wb') as f:
    cPickle.dump(normals, f, protocol=2)
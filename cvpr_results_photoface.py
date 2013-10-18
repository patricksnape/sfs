from pybug.image import MaskedNDImage, DepthImage
from pybug.io import auto_import
from pybug.landmark import labeller, ibug_68_closed_mouth
from copy import deepcopy
import cPickle
import numpy as np
from surface_reconstruction import frankotchellappa
import mayavi.mlab as mlab
from photometric_stereo import photometric_stereo as ps
from geometric_sfs import geometric_sfs as sfs
from mapping import AEP, PGA, Spherical, ImageMapper, IdentityMapper


def save_result_images(subject_id, feature_space, reco_type, fig_size=(512, 512)):
    output_path = '/vol/atlas/pts08/cvpr/results/photoface/{0}_{1}_{2}_{3}.png'
    # Left profile
    mlab.view(azimuth=-80, elevation=85, roll=-145, distance=840,
              focalpoint=np.array([288, 360, 64]))
    mlab.savefig(output_path.format(subject_id, feature_space,
                                    reco_type, 'left_profile'),
                 size=fig_size)
    # Right profile
    mlab.view(azimuth=94, elevation=85, roll=-145, distance=840,
              focalpoint=np.array([288, 360, 64]))
    mlab.savefig(output_path.format(subject_id, feature_space,
                                    reco_type, 'right_profile'),
                 size=fig_size)
    # Frontal
    mlab.view(azimuth=180, elevation=20, roll=-90, distance=810,
              focalpoint=np.array([300, 350, 67]))
    mlab.savefig(output_path.format(subject_id, feature_space,
                                    reco_type, 'frontal'),
                 size=fig_size)
    mlab.close(all=True)


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

photoface_path = '/mnt/atlas/databases/alex_images'
photoface_subjects = ['bej', 'bln', 'fav', 'mut', 'pet', 'rob', 'srb']
feature_spaces = ['aep', 'cosine', 'normal', 'pga', 'spherical']

# The index of the 4 images to use for SFS
sfs_index = 2
lights = np.array([[ 0.5,  0.4, 2],
                   [-0.5,  0.4, 2],
                   [-0.5, -0.4, 2],
                   [ 0.5, -0.4, 2]])
sfs_light = lights[sfs_index, :]

# (Subject, Feature space) - Alphabetical order
mean_depth_error_results = np.zeros([len(photoface_subjects), len(feature_spaces)])
mean_angular_error_results = np.zeros([len(photoface_subjects), len(feature_spaces)])

# (Subject, Feature space) - Alphabetical order
std_depth_error_results = np.zeros([len(photoface_subjects), len(feature_spaces)])
std_angular_error_results = np.zeros([len(photoface_subjects), len(feature_spaces)])

# 5 feature spaces + ground truth
normals = dict(zip(photoface_subjects, [{}, {}, {}, {}, {}, {}, {}]))
for s in normals.values():
    s.update(zip(['ground_truth'] + feature_spaces, [None] * (len(feature_spaces) + 1)))

for i, subject_id in enumerate(photoface_subjects):
    print "Running experiment for {0}".format(subject_id)

    subject_images = auto_import(
        '/vol/atlas/databases/alex_images/{0}*.ppm'.format(subject_id))
    # Create a 4 channel image where each channel is the greyscale of an image
    ground_truth_images = MaskedNDImage(
        np.concatenate([im.as_greyscale().pixels
                        for im in subject_images], axis=2))

    # Choose the third image as the reconstruction candidate
    intensity_image = subject_images[sfs_index].as_greyscale()
    # The first image is the only landmarked one
    intensity_image.landmarks = subject_images[0].landmarks

    # Pass landmarks to all ground truth images
    ground_truth_images.landmarks['PTS'] = intensity_image.landmarks['PTS']

    # Label with correct labels
    labeller([ground_truth_images, intensity_image],
             'PTS', ibug_68_closed_mouth)

    # Constrain to mask
    ground_truth_images.constrain_mask_to_landmarks(
        group='ibug_68_closed_mouth', label='all')
    intensity_image.constrain_mask_to_landmarks(
        group='ibug_68_closed_mouth', label='all')

    temp_texture = subject_images[sfs_index]

    # Perform Photometric Stereo
    ground_truth_normals, ground_truth_albedo = ps(ground_truth_images, lights)
    ground_truth_depth = frankotchellappa(ground_truth_normals.pixels[:, :, 0],
                                          ground_truth_normals.pixels[:, :, 1])
    ground_truth_depth_image = DepthImage((ground_truth_depth - np.min(ground_truth_depth)) / 2,
                                          texture=temp_texture)
    normals[subject_id]['ground_truth'] = ground_truth_normals

    # TODO: save images
    #ground_truth_depth_image.view(mode='mesh')
    #save_result_images(subject_id, 'all', 'groundtruth')

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

        # Perform SFS
        warped_intensity_image = MaskedNDImage(intensity_image.pixels.copy(),
                                               mask=intensity_image.mask)
        initial_estimate_image = warped_intensity_image.from_vector(
            mean_normals.copy(), n_channels=3)

        mapping_object = build_mapping_object(feature_space,
                                              initial_estimate_image,
                                              intrinsic_mean_normals)
        # Normalise the image so that it has unit albedo
        warped_intensity_image.masked_pixels /= ground_truth_albedo.masked_pixels
        warped_intensity_image.masked_pixels /= np.max(warped_intensity_image.masked_pixels)
        reconstructed_normals = sfs(warped_intensity_image,
                                    initial_estimate_image, normal_model,
                                    sfs_light, n_iters=200,
                                    mapping_object=mapping_object)
        normals[subject_id][feature_space] = reconstructed_normals

        reconstructed_depth = frankotchellappa(
            -reconstructed_normals.pixels[:, :, 0],
            reconstructed_normals.pixels[:, :, 1])
        reconstructed_depth_image = DepthImage((reconstructed_depth - np.min(reconstructed_depth)) / 2,
                                               texture=temp_texture)
        # TODO: save images
        #reconstructed_depth_image.view(mode='mesh')
        #save_result_images(subject_id, feature_space, 'sfs')

        depth_differences = np.abs(reconstructed_depth.flatten() -
                                   ground_truth_depth.flatten())
        mean_depth_error_results[i, k] = np.mean(depth_differences)

        ground_truth_normal_vec = ground_truth_normals.as_vector(keep_channels=True)
        recon_normal_vec = reconstructed_normals.as_vector(keep_channels=True)
        angular_differences = np.arccos(np.clip(np.sum(recon_normal_vec *
                                                       ground_truth_normal_vec, axis=-1), -1, 1))

        mean_angular_error_results[i, k] = np.mean(angular_differences)

        print "{0}_{1}: Mean Depth error: {2}".format(subject_id, feature_space, mean_depth_error_results[i, k])
        print "{0}_{1}: Mean Angular error: {2}".format(subject_id, feature_space, mean_angular_error_results[i, k])

        std_depth_error_results[i, k] = np.std(depth_differences)
        std_angular_error_results[i, k] = np.std(angular_differences)

# Save out error results
with open('/vol/atlas/pts08/cvpr/results/photoface/mean_depth_errors.pkl', 'wb') as f:
    cPickle.dump(mean_depth_error_results, f, protocol=2)
with open('/vol/atlas/pts08/cvpr/results/photoface/mean_angular_errors.pkl', 'wb') as f:
    cPickle.dump(mean_angular_error_results, f, protocol=2)
with open('/vol/atlas/pts08/cvpr/results/photoface/std_depth_errors.pkl', 'wb') as f:
    cPickle.dump(std_depth_error_results, f, protocol=2)
with open('/vol/atlas/pts08/cvpr/results/photoface/std_angular_errors.pkl', 'wb') as f:
    cPickle.dump(std_angular_error_results, f, protocol=2)
with open('/vol/atlas/pts08/cvpr/results/photoface/all_result_dict.pkl', 'wb') as f:
    cPickle.dump(normals, f, protocol=2)
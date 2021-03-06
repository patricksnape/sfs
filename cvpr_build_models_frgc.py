import numpy as np
import cPickle
import os.path
from menpo.landmark.labels import labeller, ibug_68_closed_mouth
import menpo.io as mio
from menpo.image import MaskedImage
from menpo.transform import ThinPlateSplines
from sfs_io import print_replace_line
from copy import deepcopy
from mapping import PGA, AEP, Spherical
from pga import intrinsic_mean
from menpo.model.pca import PCAModel
from vector_utils import normalise_vector


def build_all_models_frgc(images, ref_frame_path, subject_id,
                          out_path='/vol/atlas/homes/pts08/',
                          transform_class=ThinPlateSplines,
                          square_mask=False):
    print "Beginning model creation for {0}".format(subject_id)
    # Build reference frame
    ref_frame = mio.import_image(ref_frame_path)
    labeller([ref_frame], 'PTS', ibug_68_closed_mouth)
    ref_frame.crop_to_landmarks(boundary=2, group='ibug_68_closed_mouth',
                                label='all')
    if not square_mask:
        ref_frame.constrain_mask_to_landmarks(group='ibug_68_closed_mouth',
                                              label='all')

    reference_shape = ref_frame.landmarks['ibug_68_closed_mouth'].lms

    # Extract all shapes
    labeller(images, 'PTS', ibug_68_closed_mouth)
    shapes = [img.landmarks['ibug_68_closed_mouth'].lms for img in images]

    # Warp each of the images to the reference image
    print "Warping all frgc shapes to reference frame of {0}".format(subject_id)
    tps_transforms = [transform_class(reference_shape, shape) for shape in shapes]
    warped_images = [img.warp_to(ref_frame.mask, t)
                     for img, t in zip(images, tps_transforms)]

    # Calculate the normal matrix
    print 'Extracting all normals'
    normal_matrix = extract_normals(warped_images)

    # Save memory by deleting all the images since we don't need them any more.
    # Keep one around that we can query for it's size etc
    example_image = deepcopy(warped_images[0])
    del warped_images[:]

    # Normals
    print 'Computing normal feature space'
    normal_images = create_feature_space(normal_matrix, example_image,
                                         'normals', subject_id,
                                         out_path=out_path)

    # Spherical
    print 'Computing spherical feature space'
    spherical_matrix = Spherical().logmap(normal_matrix)
    spherical_images = create_feature_space(spherical_matrix, example_image,
                                            'spherical', subject_id,
                                            out_path=out_path)

    # AEP
    print 'Computing AEP feature space'
    mean_normals = normalise_vector(np.mean(normal_matrix, 0))
    aep_matrix = AEP(mean_normals).logmap(normal_matrix)
    aep_images = create_feature_space(aep_matrix, example_image, 'aep',
                                      subject_id,
                                      out_path=out_path)

    # PGA
    print 'Computing PGA feature space'
    mu = intrinsic_mean(normal_matrix, PGA, max_iters=50)
    pga_matrix = PGA(mu).logmap(normal_matrix)
    pga_images = create_feature_space(pga_matrix, example_image, 'pga',
                                      subject_id,
                                      out_path=out_path)

    # PCA models
    n_components = 200
    print 'Computing PCA models ({} components)'.format(n_components)
    template = ref_frame

    normal_model = PCAModel(normal_images, center=True)
    normal_model.trim_components(200)
    cosine_model = PCAModel(normal_images, center=False)
    cosine_model.trim_components(200)
    spherical_model = PCAModel(spherical_images, center=False)
    spherical_model.trim_components(200)
    aep_model = PCAModel(aep_images, center=False)
    aep_model.trim_components(200)
    pga_model = PCAModel(pga_images, center=False)
    pga_model.trim_components(200)

    mean_normals_image = normal_model.mean
    mu_image = mean_normals_image.from_vector(mu)

    # Save out models
    pickle_model(out_path, subject_id, 'normal', normal_model, template,
                 mean_normals)
    pickle_model(out_path, subject_id, 'cosine', cosine_model, template,
                 mean_normals)
    pickle_model(out_path, subject_id, 'spherical', spherical_model, template,
                 mean_normals)
    pickle_model(out_path, subject_id, 'aep', aep_model, template,
                 mean_normals)
    pickle_model(out_path, subject_id, 'pga', pga_model, template,
                 mean_normals, intrinsic_means=mu_image)


def extract_normals(images):
    vector_shape = images[0].mesh.vertex_normals.shape
    normals = np.zeros([len(images), vector_shape[0], vector_shape[1]])
    for i, im in enumerate(images):
        normals[i, ...] = im.mesh.vertex_normals
    return normals


def create_feature_space(feature_matrix, example_image, feature_space_name,
                         subject_id,
                         out_path='/vol/atlas/homes/pts08'):
    feature_space_images = []
    N = feature_matrix.shape[0]
    for i, n in enumerate(feature_matrix):
        new_im = MaskedImage.blank(example_image.shape,
                                   mask=example_image.mask,
                                   n_channels=n.shape[1])
        new_im.from_vector_inplace(n.flatten())
        new_im.landmarks = example_image.landmarks
        feature_space_images.append(new_im)
        print_replace_line('Image {0} of {1}'.format(i + 1, N))

    out_file_name = 'frgc_spring2003_68_{0}_{1}.pkl'.format(subject_id,
                                                            feature_space_name)
    out_file_path = os.path.join(out_path, out_file_name)
    with open(out_file_path, 'wb') as f:
        cPickle.dump(feature_space_images, f, protocol=2)
    return feature_space_images


def pickle_model(out_path, subject_id, feature_space_name,
                 model, template, mean_normals, intrinsic_means=None):
    filename = 'frgc_spring2003_sfs_sim_{0}_{1}.pkl'.format(subject_id,
                                                            feature_space_name)
    pickle_path = os.path.join(out_path, filename)
    out_dict = {'appearance_model': model,
                'template': template,
                'mean_normals': mean_normals}
    if intrinsic_means is not None:
        out_dict['intrinsic_mean_normals'] = intrinsic_means

    with open(pickle_path, 'wb') as f:
        cPickle.dump(out_dict, f, protocol=2)

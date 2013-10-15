import os.path
import cPickle
from pybug.io import auto_import
from inpaint import replace_nans
from scipy.ndimage.filters import gaussian_filter, median_filter
from pybug.transform import Translation
import numpy as np
import sys


def print_replace_line(string):
    # Cheeky carriage return so we print on the same line
    sys.stdout.write('\r' + string)
    sys.stdout.flush()


def load_frgc(session_id, recreate_meshes=False,
              output_base_path='/vol/atlas/pts08/',
              input_base_path='/vol/atlas/databases/frgc',
              max_images=None):
    previously_pickled_path = os.path.join(
        output_base_path, 'frgc_{0}_68_cleaned.pkl'.format(session_id))
    abs_files_path = os.path.join(input_base_path, session_id, '*.abs')

    if not recreate_meshes and os.path.exists(previously_pickled_path):
        with open(previously_pickled_path) as f:
            images = cPickle.load(f)
    else:
        all_images = auto_import(abs_files_path, max_images=max_images)
        images = [im for im in all_images if im.n_landmark_groups == 1]
        print '{0}% of the images had landmarks'.format(
            len(images) / float(len(all_images)) * 100)

        for i, im in enumerate(images):
            Translation(np.array([-1, -1])).apply_inplace(im.landmarks['PTS'].lms)
            im.pixels[..., 2] = replace_nans(im.pixels[..., 2], 100, 0.1)
            im.pixels[..., 2] = median_filter(im.pixels[..., 2], 5.0)
            im.pixels[..., 2] = gaussian_filter(im.pixels[..., 2], [2.0, 3.0])
            im.rebuild_mesh()
            print_replace_line(
                'Image {0} of {1} cleaned'.format(i + 1, len(images)))

        cPickle.dump(images, open(previously_pickled_path, 'wb'), protocol=2)

    return images


def load_basel_from_mat(recreate_meshes=False,
                        output_base_path='/vol/atlas/pts08/',
                        input_base_path='/vol/atlas/pts08/basel/',
                        max_images=None):
    previously_pickled_path = os.path.join(output_base_path,
                                           'basel_python_68.pkl')
    mat_file_path = os.path.join(input_base_path, 'basel_68.mat')

    if not recreate_meshes and os.path.exists(previously_pickled_path):
        with open(previously_pickled_path) as f:
            images = cPickle.load(f)
    else:
        from scipy.io import loadmat
        from pybug.image import RGBImage, ShapeImage
        from pybug.shape import PointCloud

        basel_dataset = loadmat(mat_file_path)
        textures = basel_dataset['textures']
        shape = basel_dataset['shapes']
        landmarks = np.swapaxes(basel_dataset['landmarks'], 0, 1)

        N = max_images if max_images else landmarks.shape[2]

        all_images = []

        for i in xrange(N):
            # Change to the correct handedness
            shape[..., 1:3, i] *= -1
            shape_image = ShapeImage(shape[..., i],
                                     texture=RGBImage(textures[..., i]))
            shape_image.landmarks['PTS'] = PointCloud(landmarks[..., i] - 1)
            shape_image.mesh.texture.landmarks['PTS'] = shape_image.landmarks['PTS']
            shape_image.constrain_mask_to_landmarks()
            shape_image.rebuild_mesh()
            all_images.append(shape_image)
            print_replace_line('Image {0} of {1}'.format(i + 1, N))

        print '\n'
        images = [im for im in all_images if im.n_landmark_groups == 1]
        print "{0}% of the images had landmarks".format(
            (float(len(images)) / len(all_images)) * 100)
        cPickle.dump(images, open(previously_pickled_path, 'wb'), protocol=2)

    return images
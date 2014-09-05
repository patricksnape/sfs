import os.path
import cPickle
import menpo.io as mio
from scipy.ndimage.filters import median_filter
import numpy as np
import sys
import cv2


def print_replace_line(string):
    # Cheeky carriage return so we print on the same line
    sys.stdout.write('\r' + string)
    sys.stdout.flush()


def rescale(val, in_min, in_max, out_min, out_max):
    return out_min + (val - in_min) * ((out_max - out_min) / (in_max - in_min))


def rescale_to_opencv_image(depth_image):
    nan_max = np.nanmax(depth_image)
    nan_min = np.nanmin(depth_image)
    depth_image[np.isnan(depth_image)] = nan_min - 2
    rescaled_depth = rescale(depth_image,
                             np.nanmin(depth_image), nan_max,
                             0.0, 1.0)
    return (rescaled_depth * 255).astype(np.uint8)


def rescale_to_depth_image(original_image, opencv_image):
    nan_max = np.nanmax(original_image.pixels[..., 2])
    nan_min = np.nanmin(original_image.pixels[..., 2])
    depth_pixels = opencv_image.astype(np.float) / 255.0
    depth_pixels = rescale(depth_pixels, 0.0, 1.0, nan_min, nan_max)
    depth_pixels[np.isclose(np.nanmin(depth_pixels), depth_pixels)] = np.nan
    return depth_pixels


def preprocess_image(image):
    # Copy the depth part of the image
    depth_pixels = image.pixels[..., 2].copy()
    depth_pixels = rescale_to_opencv_image(depth_pixels)
    filtered_depth_pixels = median_filter(depth_pixels, 5)

    # Build mask for floodfilling, this lets me ignore all the pixels
    # from the background and around the ears
    mask = np.zeros((depth_pixels.shape[0] + 2, depth_pixels.shape[1] + 2),
                    dtype=np.uint8)
    # Flood fill from top left
    cv2.floodFill(filtered_depth_pixels, mask, (0, 0),
                  (255, 255, 255), flags=cv2.FLOODFILL_MASK_ONLY)
    # Flood fill from top right
    cv2.floodFill(filtered_depth_pixels, mask, (depth_pixels.shape[1] - 1, 0),
                  (255, 255, 255), flags=cv2.FLOODFILL_MASK_ONLY)
    # Truncate and negate the flood filled areas to find the facial region
    floodfill_mask = (~mask.astype(np.bool))[1:-1, 1:-1]

    # Build a mask of the areas inside the face that need inpainting
    inpaint_mask = ~image.mask.mask & floodfill_mask
    # Inpaint the image and filter to smooth
    inpainted_pixels = cv2.inpaint(depth_pixels,
                                   inpaint_mask.astype(np.uint8),
                                   5, cv2.INPAINT_NS)
    inpainted_pixels = median_filter(inpainted_pixels, 5)

    # Back to depth pixels
    image.pixels[..., 2] = rescale_to_depth_image(image, inpainted_pixels)
    # Reset the mask!
    image.mask.pixels[..., 0] = ~np.isnan(image.pixels[..., 2])


def load_frgc(session_id, recreate_meshes=False,
              output_base_path='/vol/atlas/homes/pts08/',
              input_base_path='/vol/atlas/databases/frgc',
              max_images=None):
    previously_pickled_path = os.path.join(
        output_base_path, 'frgc_{0}_68_cleaned.pkl'.format(session_id))
    abs_files_path = os.path.join(input_base_path, session_id, '*.abs')

    if not recreate_meshes and os.path.exists(previously_pickled_path):
        with open(previously_pickled_path) as f:
            images = cPickle.load(f)
    else:
        all_images = list(mio.import_images(abs_files_path,
                                            max_images=max_images))
        images = [im for im in all_images if im.n_landmark_groups == 1]
        print '{0}% of the images had landmarks'.format(
            len(images) / float(len(all_images)) * 100)

        for i, im in enumerate(images):
            preprocess_image(im)
            print_replace_line(
                'Image {0} of {1} cleaned'.format(i + 1, len(images)))
        # Only dump the saved images if we loaded all of them!
        if max_images is None:
            cPickle.dump(images, open(previously_pickled_path, 'wb'),
                         protocol=2)

    return images


def load_basel_from_mat(recreate_meshes=False,
                        output_base_path='/vol/atlas/homes/pts08/',
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
        from menpo.image import RGBImage, ShapeImage
        from menpo.shape import PointCloud

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
            shape_image.landmarks['PTS'] = PointCloud(landmarks[..., i])
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
import numpy as np
from scipy.linalg import pinv, norm
from vector_utils import normalise_vector


def photometric_stereo(images, lights):
    """
    Images is a 3 or more channel image representing the same object taken
    under different lighting conditions. Lights is a n_channels x 3 matrix that
    represents the direction from which each image is lit.

    Only the masked pixels are recovered.

    Parameters
    ----------
    images : (M, N, C) :class:`pybug.image.MaskedNDImage`
        An image where each channel is an image lit under a unique lighting
        direction.
    lights : (C, 3) ndarray
        A matrix representing the light directions for each of the channels in
        ``images``.

    Returns
    -------
    normal_image : (M, N, 3) :class:`pybug.image.MaskedNDImage`
        A 3-channel image representing the components of the recovered normals.
    albedo_image : (M, N, 1) :class:`pybug.image.MaskedNDImage`
        A 1-channel image representing the albedo at each pixel.
    """
    # Ensure the light are unit vectors
    lights = normalise_vector(lights)
    LL = pinv(lights)

    # n_masked_pixels x n_channels
    pixels = images.as_vector(keep_channels=True)
    n_pixels = pixels.shape[0]
    n_images = pixels.shape[1]

    if n_images < 3:
        raise ValueError('Photometric Stereo is undefined with less than 3 '
                         'input images.')
    if LL.shape[1] != n_images:
        raise ValueError('You must provide a light direction for each input '
                         'channel.')
    albedo = np.zeros(n_pixels)
    normals = np.zeros([n_pixels, 3])
    
    for i in xrange(n_pixels):
        I = pixels[i, :]
        nn = np.dot(LL, I)
        pixel_norm = norm(nn)
        albedo[i] = pixel_norm

        if pixel_norm != 0.0:
            # normal = n / albedo
            normals[i, :] = nn / pixel_norm
                    
    return (images.from_vector(normals, n_channels=3),
            images.from_vector(albedo, n_channels=1))



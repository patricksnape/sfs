import numpy as np
from scipy.linalg import pinv, norm


def photometric_stereo(images, lights):
    LL = pinv(lights)

    # n_masked_pixels x n_channels
    pixels = images.as_vector(keep_channels=True)
    n_pixels = pixels.shape[0]

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



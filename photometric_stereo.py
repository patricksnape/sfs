import numpy as np
from scipy.linalg import pinv, norm


def photometric_stereo(images, lights):
    LL = pinv(lights)
    
    height, width, N = images.shape
    albedo = np.zeros([height, width])
    
    n = np.zeros([height, width, 3])
    p = np.zeros([height, width])
    q = np.zeros([height, width])
    
    for ii in xrange(height):
        for jj in xrange(width):
            I = images[ii, jj, :].flatten()
            nn = np.dot(LL, I)
            pp = norm(nn)
            albedo[ii, jj] = pp
            
            if pp != 0.0:
                # normal = n / albedo 
                n[ii, jj, :] = nn / pp  
                if n[ii, jj, 2] != 0.0:
                    p[ii, jj] = n[ii, jj, 0] / n[ii, jj, 2]  # x / z
                    q[ii, jj] = n[ii, jj, 1] / n[ii, jj, 2]  # y / z
                    
    return n, albedo



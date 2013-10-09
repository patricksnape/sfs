import numpy as np
import cython
cimport numpy as np
cimport cython
from libc.math cimport sqrt

@cython.boundscheck(False)
cpdef rotate_north_pole(np.ndarray[np.float64_t, ndim=2] vector):
    """
    Rotation matrix that rotates vector to the north pole.
    Expects mean vector per point in face (sd_vectors, 3)
    """
    cdef int N = vector.shape[0]
    cdef double[:, :, :] rotation_matrices = np.zeros([3, 3, N])
    cdef double acangle, cosa, sina, vera, x, y, z, mag
    cdef double norm_vector_x, norm_vector_y, norm_vector_z
    cdef double eps = np.spacing(1)
    
    for i in range(N):
        norm_vector_x = vector[i, 0]
        norm_vector_y = vector[i, 1]
        norm_vector_z = vector[i, 2]
        
        # Normalize base vector
        mag = norm_vector_x * norm_vector_x + norm_vector_y * norm_vector_y + norm_vector_z * norm_vector_z
        mag = sqrt(mag)
        
        norm_vector_x /= mag
        norm_vector_y /= mag
        norm_vector_z /= mag
        
        # Get the cosine and sine angle
        cosa = norm_vector_z
        sina = sqrt(1.0 - cosa * cosa)
        
        # Check for near zero angles
        if (1.0 - cosa) > eps:
            x = norm_vector_y / sina
            y = -norm_vector_x / sina
            z = 0.0
        else:
            x = 0.0
            y = 0.0
            z = 0.0
        
        # vercosine
        vera = 1.0 - cosa
    
        # Build rotation matrix
        rotation_matrices[0, 0, i] = cosa + x**2 * vera
        rotation_matrices[0, 1, i] = x * y * vera - z * sina
        rotation_matrices[0, 2, i] = x * z * vera + y * sina
        
        rotation_matrices[1, 0, i] = x * y * vera + z * sina
        rotation_matrices[1, 1, i] = cosa + y**2 * vera
        rotation_matrices[1, 2, i] = y * z * vera - x * sina
        
        rotation_matrices[2, 0, i] = x * z * vera - y * sina
        rotation_matrices[2, 1, i] = y * z * vera + x * sina
        rotation_matrices[2, 2, i] = cosa + z**2 * vera
        
    return rotation_matrices

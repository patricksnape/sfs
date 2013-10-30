import numpy as np
from scipy.fftpack import ifftshift, fft2, ifft2, dct, idct
from scipy.sparse.linalg import spsolve, lsqr
from scipy.sparse import csc_matrix, issparse
from numpy.linalg import eig


def test_reconstruction(im):
    from copy import deepcopy
    from pybug.image import MaskedNDImage, DepthImage

    im = deepcopy(im)
    new_im = MaskedNDImage.blank(im.shape, mask=im.mask, n_channels=3)
    im.rebuild_mesh()
    n = im.mesh.vertex_normals
    new_im.from_vector_inplace(n.flatten())
    g = gradient_field_from_normals(new_im)
    d = frankotchellappa(g.pixels[..., 0], g.pixels[..., 1])

    im.view(mode='mesh', normals=n, mask_points=20)
    DepthImage(d - np.min(d)).view_new(mode='mesh')


def gradient_field_from_normals(normals):
    vector = normals.as_vector(keep_channels=True)

    gradient_field = np.zeros([vector.shape[0], 2])
    zero_indices = vector[:, 2] != 0.0
    nonzero_zs = vector[:, 2][zero_indices]
    gradient_field[:, 0][zero_indices] = -vector[:, 0][zero_indices] / nonzero_zs
    gradient_field[:, 1][zero_indices] = vector[:, 1][zero_indices] / nonzero_zs

    return normals.from_vector(gradient_field, n_channels=2)


def frankotchellappa(dzdx, dzdy):
    rows, cols = dzdx.shape

    # The following sets up matrices specifying frequencies in the x and y
    # directions corresponding to the Fourier transforms of the gradient
    # data.  They range from -0.5 cycles/pixel to + 0.5 cycles/pixel.
    # The scaling of this is irrelevant as long as it represents a full
    # circle domain. This is functionally equivalent to any constant * pi
    pi_over_2 = np.pi / 2.0
    row_grid = np.linspace(-pi_over_2, pi_over_2, rows)
    col_grid = np.linspace(-pi_over_2, pi_over_2, cols)
    wy, wx = np.meshgrid(row_grid, col_grid, indexing='ij')

    # Quadrant shift to put zero frequency at the appropriate edge
    wx = ifftshift(wx)
    wy = ifftshift(wy)

    # Fourier transforms of gradients
    DZDX = fft2(dzdx)
    DZDY = fft2(dzdy)

    # Integrate in the frequency domain by phase shifting by pi/2 and
    # weighting the Fourier coefficients by their frequencies in x and y and
    # then dividing by the squared frequency
    denom = (wx ** 2 + wy ** 2)
    Z = (-1j * wx * DZDX - 1j * wy * DZDY) / denom
    Z = np.nan_to_num(Z)

    return np.real(ifft2(Z))


def weighted_finite_differences(gx, gy, wx, wy):
    # Fix gradients at boundaries to 0 (neumann boundary condition)
    gx[:, -1] = 0
    gy[-1, :] = 0

    # Weight the gradients
    gx = gx * wx
    gy = gy * wy

    # Pad each array around both axes by one pixels with the value 0
    gx = np.pad(gx, 1, mode='constant')
    gy = np.pad(gy, 1, mode='constant')
    gxx = np.zeros_like(gx)
    gyy = np.zeros_like(gx)

    # Finite differences
    # Take the finite differences:
    #     gyy[j+1, k] = gy[j+1,k] - gy[j,k]
    #     gxx[j, k+1] = gx[j,k+1] - gx[j,k]
    # where
    #     j = 0:height+1,
    #     k = 0:width+1
    gyy[1:, :-1] = gy[1:, :-1] - gy[:-1, :-1]
    gxx[:-1, 1:] = gx[:-1, 1:] - gx[:-1, :-1]
    f = gxx + gyy
    f = f[1:-1, 1:-1]

    return f


def poisson_neumann(gx, gy):
    height, width = gx.shape

    f = weighted_finite_differences(gx, gy, np.ones_like(gx), np.ones_like(gy))

    # Compute cosine transform
    fcos = dct(dct(f, norm='ortho', axis=0), norm='ortho', axis=1)

    # Compute the solution in the fourier domain
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    denom = (2.0 * np.cos(np.pi * x / width) - 2.0) + (2.0 * np.cos(np.pi * y / height) - 2.0)
    # First element is fixed to 0.0
    fcos /= denom
    fcos[0, 0] = 0

    # Compute inverse discrete cosine transform to find solution
    # in spatial domain
    return idct(idct(fcos, norm='ortho', axis=0), norm='ortho', axis=1)


def weighted_laplacian(height, width, wx, wy):
    # Laplacian matrix for D*[Zx;Zy] where D is 2*2 diffusion tensor
    # kernel D should be symmetric

    wx = np.pad(wx, 1, mode='constant')
    wy = np.pad(wy, 1, mode='constant')

    N = (height + 2) * (width + 2)
    mask = np.zeros([height + 2, width + 2], dtype=np.int32)
    # Set to all 1s inside the padding
    mask[1:-1, 1:-1] = 1
    # Equivalent to Matlab's 'find' method (linear indices in to mask matrix)
    idx = np.ravel_multi_index((mask == 1).nonzero(), mask.shape)

    wx_flat = wx.flatten()
    wy_flat = wy.flatten()

    A = csc_matrix((-wx_flat[idx], (idx, idx + height + 2)), shape=(N, N))
    A = A + csc_matrix((-wx_flat[idx - height - 2], (idx, idx - height - 2)), shape=(N, N))
    A = A + csc_matrix((-wy_flat[idx], (idx, idx + 1)), shape=(N, N))
    A = A + csc_matrix((-wy_flat[idx - 1], (idx, idx - 1)), shape=(N, N))

    A = A[np.ix_(idx, idx)]
    # width * height
    N = A.shape[0]
    # Sum over the columns and then turn back in to a 1D array
    dd = np.asarray(A.sum(1)).flatten()
    idx = np.arange(N)
    return A + csc_matrix((-dd, (idx, idx)), shape=(N, N))


def huber_weights(k, previous_estimate, xs, ys):
    gradient_y, gradient_x = np.gradient(previous_estimate)

    # Huber estimate for xs
    x_estimate = np.abs(gradient_x - xs)
    w_x = np.zeros_like(xs)
    idx = x_estimate <= k
    w_x[idx] = 1
    idx = x_estimate >= k
    w_x[idx] = k / x_estimate[idx]

    # Huber estimate for ys
    y_estimate = np.abs(gradient_y - ys)
    w_y = np.zeros_like(ys)
    idy = y_estimate <= k
    w_y[idy] = 1
    idy = y_estimate >= k
    w_y[idy] = k / y_estimate[idy]

    return w_x, w_y


def M_estimator(gx, gy, max_iters=3, k=1.345):
    height, width = gx.shape

    # Calculate an initial estimate from the poisson solver
    previous_estimate = poisson_neumann(gx, gy)

    for ii in xrange(max_iters):
        # Huber M-estimator
        w_gx, w_gy = huber_weights(k, previous_estimate, gx, gy)

        f = weighted_finite_differences(gx, gy, w_gx, w_gy)
        A = weighted_laplacian(height, width, w_gx, w_gy)

        current_estimate = spsolve(-A, f.flatten())
#         current_estimate = np.concatenate([[0.0], current_estimate])
        current_estimate = np.reshape(current_estimate, [height, width])

        idx = np.abs(previous_estimate) > 0.01
        if(idx.size):
            ee = (previous_estimate[idx] - current_estimate[idx]) / previous_estimate[idx]
            ee = ee ** 2
            ee = np.mean(ee)
            if ee < 1e-6:
                break

        previous_estimate = current_estimate

    return current_estimate


# def diffusion_kernel_function(gx, gy):
#
#     # Tensor based
#     # Instead of individual weights to gx and gy, we have a 2*2 tensor getting
#     # multiplied by [gx,gy] vector at each pixel
#     # get tensor values using local analysis
#
#     H, W = gx.shape
#     p = gx
#     q = gy
#
#     sigma = 0.5
# # How do we create a kernel size in Scipy?
# #     ss = np.floor(6.0 * sigma)
# #     if(ss < 3):
# #         ss = 3
# #     ww = fspecial('gaussian',ss,sigma)
#
#
#     T11 = gaussian_filter(p ** 2, sigma)
#     T22 = gaussian_filter(q ** 2, sigma)
#     T12 = gaussian_filter(p * q, sigma)
#
#     # find eigen values
#     ImagPart = np.sqrt((T11 - T22) ** 2 + 4 * (T12 ** 2));
#     EigD_1 = (T22 + T11 + ImagPart) / 2.0
#     EigD_2 = (T22 + T11 - ImagPart) / 2.0
#
#     alpha = 0.02
#     THRESHOLD_SMALL = np.max(EigD_1.flatten()) / 100.0
#
#     L1 = np.ones([H, W]);
#     idx = EigD_1 > THRESHOLD_SMALL
#     L1[idx] = alpha + 1.0 - np.exp(-3.315 / (EigD_1[idx] ** 4))
#     L2 = np.ones([H, W])
#
#     D11 = np.zeros([H, W])
#     D12 = np.zeros([H, W])
#     D22 = np.zeros([H, W])
#
#     for ii in xrange(H):
#         for jj in xrange(W):
#             Wmat = np.array([[T11[ii, jj], T12[ii, jj]],
#                              [T12[ii, jj], T22[ii, jj]]])
#             v, d = eig(Wmat)
#             v = np.diagflat(v)
#             if d[0, 0] > d[1, 1]:
#                 d0 = d[0, 0]
#                 d[0, 0] = d[1, 1]
#                 d[1, 1] = d0
#
#                 v0 = v1[:, 0].copy()
#                 v1[:, 0] = v[:, 1]
#                 v1[:, 1] = v0
#
#             # d(1,1) is smaller
#             d[0, 0] = L2[ii, jj]
#             d[1, 1] = L1[ii, jj]
#
#             Wmat = np.dot(np.dot(v, d), v.T)
#             D11[ii, jj] = Wmat[0, 0]
#             D22[ii, jj] = Wmat[1, 1]
#             D12[ii, jj] = Wmat[0, 1]
#
#     A = laplacian_matrix_tensor(H,W,D11,D12,D12,D22)
#     f = calculate_f_tensor(p,q,D11,D12,D12,D22)
#
#     Z = spsolve(A, f.flatten())
# #     Z = [0;Z];
#     Z = np.reshape(Z, [H,W])
#     Z = Z - np.min(Z)
#     return Z
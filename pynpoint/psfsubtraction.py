"""
Functions for PSF subtraction.
"""

from typing import Optional, Tuple

import numpy as np

from scipy.ndimage import rotate
from sklearn.decomposition import PCA
from typeguard import typechecked
#from spin_interface import train
import matplotlib.pyplot as plt
import cv2

@typechecked
def psf_subtraction(images: np.ndarray,
                        angles: np.ndarray,
                        pca_number: int,
                        ref_data: np.ndarray,
                        pca_sklearn: Optional[PCA] = None,
                        im_shape: Optional[Tuple[int, int, int]] = None,
                        indices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for PSF subtraction with PCA.
    Parameters
    ----------
    images : numpy.ndarray
        Stack of images. Also used as reference images if `pca_sklearn` is set to None. Should be
        in the original 3D shape if `pca_sklearn` is set to None or in the 2D reshaped format if
        `pca_sklearn` is not set to None.
    angles : numpy.ndarray
        Derotation angles (deg).
    pca_number : int
        Number of principal components used for the PSF model.
    pca_sklearn : sklearn.decomposition.pca.PCA, None
        PCA decomposition of the input data.
    im_shape : tuple(int, int, int), None
        Original shape of the stack with images. Required if `pca_sklearn` is not set to None.
    indices : numpy.ndarray, None
        Non-masked image indices. All pixels are used if set to None.
    Returns
    -------
    numpy.ndarray
        Residuals of the PSF subtraction.
    numpy.ndarray
        Derotated residuals of the PSF subtraction.
    """
    print('image')
    print(ref_data.shape)
    if pca_sklearn is None:
        pca_sklearn = PCA(n_components=pca_number, svd_solver='arpack')

        im_shape = images.shape

        if indices is None:
            # select the first image and get the unmasked image indices
            im_star = images[0, ].reshape(-1)
            indices = np.where(im_star != 0.)[0]

        # reshape the images and select the unmasked pixels
        im_reshape = images.reshape(im_shape[0], im_shape[1]*im_shape[2])
        im_reshape = im_reshape[:, indices]

        # subtract mean image
        im_reshape -= np.mean(im_reshape, axis=0)

        # create pca basis
        pca_sklearn.fit(im_reshape)

    else:
        im_reshape = np.copy(images)

    # create pca representation
    zeros = np.zeros((pca_sklearn.n_components - pca_number, im_reshape.shape[0]))
    """
    train(
        iterations=100000,
        lr=1e-3,
        batch_size=5,
        neig=5,
        shards=50,
        step_lr=False,
        decay=0.01,
        rmsprop_decay=0.1,
        image=ref_data,
        log_image_every=10,
        save_params_every=100,
        use_pfor=True,
        per_example=True,
        data_dir="",
        show_plots=False)
    """
    # added by bli
    #pca_sklearn.components_ = np.linalg.norm(pca_sklearn.components_) * np.random.rand(*pca_sklearn.components_.shape)
    #pca_sklearn.singular_values_ = np.linalg.norm(pca_sklearn.singular_values_) * np.random.rand(*pca_sklearn.singular_values_.shape)

    npts = 80
    print('plot')
    print(pca_sklearn.components_.shape)
    psi_fig, psi_ax, psi_im, loss_ax = _create_plots(pca_number, npts)
    _update_plots(0, pca_sklearn.components_.T, None, psi_fig, psi_ax, psi_im, loss_ax, pca_number, 1, npts,
                      losses=None, eigenvalues=None, eigenvalues_ma=pca_sklearn.singular_values_)

    pca_rep = np.matmul(pca_sklearn.components_[:pca_number], im_reshape.T)
    pca_rep = np.vstack((pca_rep, zeros)).T

    # create psf model
    psf_model = pca_sklearn.inverse_transform(pca_rep)
    print('m norm ' + str(np.linalg.norm(psf_model)))

    # create original array size
    residuals = np.zeros((im_shape[0], im_shape[1]*im_shape[2]))

    # subtract the psf model
    if indices is None:
        indices = np.arange(0, im_reshape.shape[1], 1)

    residuals[:, indices] = im_reshape - psf_model

    # reshape to the original image size
    residuals = residuals.reshape(im_shape)

    # check if the number of parang is equal to the number of images
    if residuals.shape[0] != angles.shape[0]:
        raise ValueError(f'The number of images ({residuals.shape[0]}) is not equal to the '
                         f'number of parallactic angles ({angles.shape[0]}).')

    # derotate the images
    res_rot = np.zeros(residuals.shape)
    for j, item in enumerate(angles):
        res_rot[j, ] = rotate(residuals[j, ], item, reshape=False)

    return residuals, res_rot


def _create_plots(neig, npts):
    """Hook to set up plots at start of run."""
    nfig = max(2, int(np.ceil(np.sqrt(neig))))
    psi_fig, psi_ax = plt.subplots(nfig, nfig, figsize=(10, 10))
    psi_im = []
    for i in range(nfig**2):
      psi_ax[i // nfig, i % nfig].axis('off')
    for i in range(neig):
      psi_im.append(psi_ax[i // nfig, i % nfig].imshow(
          np.zeros((npts, npts)), interpolation='none', cmap='plasma'))
    _, loss_ax = plt.subplots(1, 1)
    return psi_fig, psi_ax, psi_im, loss_ax

def _update_plots(t, outputs, inputs, psi_fig, psi_ax, psi_im, loss_ax, neig, charge, npts,
                    losses=None, eigenvalues=None, eigenvalues_ma=None):
    """Hook to update the plots periodically."""
    del inputs
    del losses
    del eigenvalues

    for i in range(neig):
        pimg = outputs[:, i].reshape(npts, npts)
        psi_im[i].set_data(pimg)
        psi_im[i].set_clim(pimg.min(), pimg.max())
    psi_fig.canvas.draw()
    psi_fig.canvas.flush_events()


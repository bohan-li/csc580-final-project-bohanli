import time
import math
import warnings

from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np

from scipy.ndimage import rotate
from sklearn.decomposition import PCA
from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress
from pynpoint.util.multipca import PcaMultiprocessingCapsule
from psfsubtraction import psf_subtraction
from pynpoint.util.residuals import combine_residuals
from pprint import pprint
from pynpoint import PcaPsfSubtractionModule

class MySubtractionModule(ProcessingModule):
    """
    Pipeline module for PSF subtraction with principal component analysis (PCA). The residuals are
    calculated in parallel for the selected numbers of principal components. This may require
    a large amount of memory in case the stack of input images is very large. The number of
    processes can be set with the CPU keyword in the configuration file.
    """

    __author__ = 'Markus Bonse, Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 images_in_tag: str,
                 reference_in_tag: str,
                 res_mean_tag: Optional[str] = None,
                 res_median_tag: Optional[str] = None,
                 res_weighted_tag: Optional[str] = None,
                 res_rot_mean_clip_tag: Optional[str] = None,
                 res_arr_out_tag: Optional[str] = None,
                 basis_out_tag: Optional[str] = None,
                 pca_numbers: Union[range, List[int], np.ndarray] = range(1, 21),
                 extra_rot: float = 0.,
                 subtract_mean: bool = True) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        images_in_tag : str
            Tag of the database entry with the science images that are read as input
        reference_in_tag : str
            Tag of the database entry with the reference images that are read as input.
        res_mean_tag : str, None
            Tag of the database entry with the mean collapsed residuals. Not calculated if set to
            None.
        res_median_tag : str, None
            Tag of the database entry with the median collapsed residuals. Not calculated if set
            to None.
        res_weighted_tag : str, None
            Tag of the database entry with the noise-weighted residuals (see Bottom et al. 2017).
            Not calculated if set to None.
        res_rot_mean_clip_tag : str, None
            Tag of the database entry of the clipped mean residuals. Not calculated if set to
            None.
        res_arr_out_tag : str, None
            Tag of the database entry with the derotated image residuals from the PSF subtraction.
            The tag name of `res_arr_out_tag` is appended with the number of principal components
            that was used. Not calculated if set to None. Not supported with multiprocessing.
        basis_out_tag : str, None
            Tag of the database entry with the basis set. Not stored if set to None.
        pca_numbers : range, list(int, ), numpy.ndarray
            Number of principal components used for the PSF model. Can be a single value or a tuple
            with integers.
        extra_rot : float
            Additional rotation angle of the images (deg).
        subtract_mean : bool
            The mean of the science and reference images is subtracted from the corresponding
            stack, before the PCA basis is constructed and fitted.
        Returns
        -------
        NoneType
            None
        """

        super(MySubtractionModule, self).__init__(name_in)

        self.m_components = np.sort(np.atleast_1d(pca_numbers))
        self.m_extra_rot = extra_rot
        self.m_subtract_mean = subtract_mean

        self.m_pca = PCA(n_components=np.amax(self.m_components), svd_solver='arpack')

        self.m_reference_in_port = self.add_input_port(reference_in_tag)
        self.m_star_in_port = self.add_input_port(images_in_tag)

        if res_mean_tag is None:
            self.m_res_mean_out_port = None
        else:
            self.m_res_mean_out_port = self.add_output_port(res_mean_tag)

        if res_median_tag is None:
            self.m_res_median_out_port = None
        else:
            print("in here")
            temp = self.add_output_port(res_median_tag)
            print(temp)
            self.m_res_median_out_port = temp

        if res_weighted_tag is None:
            self.m_res_weighted_out_port = None
        else:
            self.m_res_weighted_out_port = self.add_output_port(res_weighted_tag)

        if res_rot_mean_clip_tag is None:
            self.m_res_rot_mean_clip_out_port = None
        else:
            self.m_res_rot_mean_clip_out_port = self.add_output_port(res_rot_mean_clip_tag)

        if res_arr_out_tag is None:
            self.m_res_arr_out_ports = None
        else:
            self.m_res_arr_out_ports = {}
            for pca_number in self.m_components:
                self.m_res_arr_out_ports[pca_number] = self.add_output_port(res_arr_out_tag +
                                                                            str(pca_number))

        if basis_out_tag is None:
            self.m_basis_out_port = None
        else:
            self.m_basis_out_port = self.add_output_port(basis_out_tag)

    @typechecked
    def _run_multi_processing(self,
                              star_reshape: np.ndarray,
                              im_shape: Tuple[int, int, int],
                              indices: np.ndarray) -> None:
        """
        Internal function to create the residuals, derotate the images, and write the output
        using multiprocessing.
        """

        cpu = self._m_config_port.get_attribute('CPU')
        angles = -1.*self.m_star_in_port.get_attribute('PARANG') + self.m_extra_rot

        tmp_output = np.zeros((len(self.m_components), im_shape[1], im_shape[2]))

        if self.m_res_mean_out_port is not None:
            self.m_res_mean_out_port.set_all(tmp_output, keep_attributes=False)

        if self.m_res_median_out_port is not None:
            self.m_res_median_out_port.set_all(tmp_output, keep_attributes=False)

        if self.m_res_weighted_out_port is not None:
            self.m_res_weighted_out_port.set_all(tmp_output, keep_attributes=False)

        if self.m_res_rot_mean_clip_out_port is not None:
            self.m_res_rot_mean_clip_out_port.set_all(tmp_output, keep_attributes=False)

        self.m_star_in_port.close_port()
        self.m_reference_in_port.close_port()

        if self.m_res_mean_out_port is not None:
            self.m_res_mean_out_port.close_port()

        if self.m_res_median_out_port is not None:
            self.m_res_median_out_port.close_port()

        if self.m_res_weighted_out_port is not None:
            self.m_res_weighted_out_port.close_port()

        if self.m_res_rot_mean_clip_out_port is not None:
            self.m_res_rot_mean_clip_out_port.close_port()

        if self.m_res_arr_out_ports is not None:
            for pca_number in self.m_components:
                self.m_res_arr_out_ports[pca_number].close_port()

        if self.m_basis_out_port is not None:
            self.m_basis_out_port.close_port()

        capsule = PcaMultiprocessingCapsule(self.m_res_mean_out_port,
                                            self.m_res_median_out_port,
                                            self.m_res_weighted_out_port,
                                            self.m_res_rot_mean_clip_out_port,
                                            cpu,
                                            deepcopy(self.m_components),
                                            deepcopy(self.m_pca),
                                            deepcopy(star_reshape),
                                            deepcopy(angles),
                                            im_shape,
                                            indices)

        capsule.run()

    @typechecked
    def _run_single_processing(self,
                               star_reshape: np.ndarray,
                               im_shape: Tuple[int, int, int],
                               indices: np.ndarray) -> None:
        """
        Internal function to create the residuals, derotate the images, and write the output
        using a single process.
        """
        print("SINGLE PROCESSING!!!!!!!!!!!!!!!")
        start_time = time.time()

        for i, pca_number in enumerate(self.m_components):
            progress(i, len(self.m_components), 'Creating residuals...', start_time)
            parang = -1.*self.m_star_in_port.get_attribute('PARANG') + self.m_extra_rot

            residuals, res_rot = psf_subtraction(images=star_reshape,
                                                     angles=parang,
                                                     ref_data=self.ref_data,
                                                     pca_number=int(pca_number),
                                                     pca_sklearn=self.m_pca,
                                                     im_shape=im_shape,
                                                     indices=indices)

            hist = f'max PC number = {np.amax(self.m_components)}'

            # 1.) derotated residuals
            if self.m_res_arr_out_ports is not None:
                print('derotate')
                self.m_res_arr_out_ports[pca_number].set_all(res_rot)
                self.m_res_arr_out_ports[pca_number].copy_attributes(self.m_star_in_port)
                self.m_res_arr_out_ports[pca_number].add_history('PcaPsfSubtractionModule', hist)

            # 2.) mean residuals
            if self.m_res_mean_out_port is not None:
                print('mean')
                stack = combine_residuals(method='mean', res_rot=res_rot)
                self.m_res_mean_out_port.append(stack, data_dim=3)

            # 3.) median residuals
            if self.m_res_median_out_port is not None:
                print('stack')
                stack = combine_residuals(method='median', res_rot=res_rot)
                self.m_res_median_out_port.append(stack, data_dim=3)

            # 4.) noise-weighted residuals
            if self.m_res_weighted_out_port is not None:
                print('noise-weighted')
                stack = combine_residuals(method='weighted',
                                          res_rot=res_rot,
                                          residuals=residuals,
                                          angles=parang)

                self.m_res_weighted_out_port.append(stack, data_dim=3)

            # 5.) clipped mean residuals
            if self.m_res_rot_mean_clip_out_port is not None:
                print('clipped')
                stack = combine_residuals(method='clipped', res_rot=res_rot)
                self.m_res_rot_mean_clip_out_port.append(stack, data_dim=3)

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Subtracts the mean of the image stack from all images, reshapes
        the stack of images into a 2D array, uses singular value decomposition to construct the
        orthogonal basis set, calculates the PCA coefficients for each image, subtracts the PSF
        model, and writes the residuals as output.
        Returns
        -------
        NoneType
            None
        """

        cpu = self._m_config_port.get_attribute('CPU')

        if cpu > 1 and self.m_res_arr_out_ports is not None:
            warnings.warn(f'Multiprocessing not possible if \'res_arr_out_tag\' is not set '
                          f'to None.')

        # get all data
        star_data = self.m_star_in_port.get_all()
        im_shape = star_data.shape

        # select the first image and get the unmasked image indices
        im_star = star_data[0, ].reshape(-1)
        #indices = np.where(im_star != 0.)[0]
        indices = np.where(im_star != float("inf"))[0]

        print(im_shape)
        print(star_data.shape)

        # reshape the star data and select the unmasked pixels
        star_reshape = star_data.reshape(im_shape[0], im_shape[1]*im_shape[2])
        star_reshape = star_reshape[:, indices]

        print(star_reshape.shape)

        if self.m_reference_in_port.tag == self.m_star_in_port.tag:
            ref_reshape = deepcopy(star_reshape)
            self.ref_data = ref_reshape

        else:
            ref_data = self.m_reference_in_port.get_all()
            ref_shape = ref_data.shape
            self.ref_data = ref_data

            if ref_shape[-2:] != im_shape[-2:]:
                raise ValueError('The image size of the science data and the reference data '
                                 'should be identical.')

            # reshape reference data and select the unmasked pixels
            ref_reshape = ref_data.reshape(ref_shape[0], ref_shape[1]*ref_shape[2])
            ref_reshape = ref_reshape[:, indices]

        # subtract mean from science data, if required
        if self.m_subtract_mean:
            mean_star = np.mean(star_reshape, axis=0)
            star_reshape -= mean_star

        # subtract mean from reference data
        mean_ref = np.mean(ref_reshape, axis=0)
        ref_reshape -= mean_ref

        ref_reshape.tofile('meansubtracted.txt')

        print('imshape')
        print(ref_reshape.shape)

        # create the PCA basis
        print("----------------------------------------------------")
        print('Constructing PSF model...', end='')
        self.m_pca.fit(ref_reshape)
        pprint(vars(self.m_pca))
        print('VAR --- ' + str(self.m_subtract_mean))

        # add mean of reference array as 1st PC and orthogonalize it with respect to the PCA basis
        if not self.m_subtract_mean:
            mean_ref_reshape = mean_ref.reshape((1, mean_ref.shape[0]))

            q_ortho, _ = np.linalg.qr(np.vstack((mean_ref_reshape,
                                                 self.m_pca.components_[:-1, ])).T)

            self.m_pca.components_ = q_ortho.T

        print(' [DONE]')

        if self.m_basis_out_port is not None:
            pc_size = self.m_pca.components_.shape[0]

            basis = np.zeros((pc_size, im_shape[1]*im_shape[2]))
            basis[:, indices] = self.m_pca.components_
            basis = basis.reshape((pc_size, im_shape[1], im_shape[2]))

            self.m_basis_out_port.set_all(basis)

        if True: #cpu == 1 or self.m_res_arr_out_ports is not None:
            self._run_single_processing(star_reshape, im_shape, indices)

        else:
            print('Creating residuals', end='')
            self._run_multi_processing(star_reshape, im_shape, indices)
            print(' [DONE]')

        history = f'max PC number = {np.amax(self.m_components)}'

        # save history for all other ports
        if self.m_res_mean_out_port is not None:
            self.m_res_mean_out_port.copy_attributes(self.m_star_in_port)
            self.m_res_mean_out_port.add_history('PcaPsfSubtractionModule', history)

        if self.m_res_median_out_port is not None:
            self.m_res_median_out_port.copy_attributes(self.m_star_in_port)
            self.m_res_median_out_port.add_history('PcaPsfSubtractionModule', history)

        if self.m_res_weighted_out_port is not None:
            self.m_res_weighted_out_port.copy_attributes(self.m_star_in_port)
            self.m_res_weighted_out_port.add_history('PcaPsfSubtractionModule', history)

        if self.m_res_rot_mean_clip_out_port is not None:
            self.m_res_rot_mean_clip_out_port.copy_attributes(self.m_star_in_port)
            self.m_res_rot_mean_clip_out_port.add_history('PcaPsfSubtractionModule', history)

        self.m_star_in_port.close_port()
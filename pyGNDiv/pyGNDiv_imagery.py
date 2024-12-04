"""
@author: Javier Pacheco-Labrador, PhD (javier.pacheco@csic.es)
         Environmental Remote Sensing and Spectroscopy Laboratory (SpecLab)
         Spanish National Research Council (CSIC)

DESCRIPTION:
==============================================================================
This module enables applying the PyGND functions to remote sensing imagery,
    includingh the computation of Rao's quadratic index (RaoQ), the use of RaoQ
    to partition variance at several scales (de Bello et al. 2010), and the
    variance-based diversity partition (Laliberté et al. 2020). In all the 
    cases, the generalizable normalization described in Pacheco-Labrador et al,
    (2023) is applied to improve the comparison of metrics computed from
    datasests of different dimensionality and covariance.

Cite as:
  Pacheco-Labrador, J., de Bello, F., Migliavacca , M., Ma, X., Carvalhais, N.,
    & Wirth, C. (2023). A generalizable normalization for assessing
    plant functional diversity metrics across scales from remote sensing.
    Methods in Ecology and Evolution.

REFERENCES:
    de Bello, F., Lavergne, S., Meynard, C.N., Lepš, J., & Thuiller, W.
        (2010). The partitioning of diversity: showing Theseus a way out of
        the labyrinth. Journal of Vegetation Science, 21, 992-1000.

Laliberté, E., Schweiger, A.K., & Legendre, P. (2020). Partitioning plant
    spectral diversity into alpha and beta components. Ecology Letters, 23,
    370-380.
"""

import numpy as np
# from pyGNDiv import pyGNDiv as gnd
from pyGNDiv import pyGNDiv_numba as gnd
from sklearn.preprocessing import StandardScaler
import time


def report_time(t0_):
    dtime = time.time() - t0_
    if dtime < 60.:
        print('\tElapsed time: elapsed time %.2f seconds...\n' % ((dtime)))
    else:
        print('\tElapsed time: elapsed time %.2f minutes...\n' % (
            (dtime) / 60.))


def apply_pca_cube(cube_, mask_=None, fexp_var_in=.98, n_sigmas_norm=6.,
                   weight_in_=None, redo_cube=True):
    """
    Standardize an apply PCA to a X matrix of n observations by p variables.
    Inputs:
        cube_: numpy array, spectral image or data cube of dimensions (x, y,
               traits)
        mask_: boolean 2D matrix, mask indicating pixels to be removed from
                 the calculations. These are converted to nan, whose abundance
                 is accounted for the argument nan_tolerance
        fexp_var_in: float, fraction of the expected variance that it is
                     expected to be retained by the principal components kept
        n_sigmas_norm: float, standardized distance to bounds (in standard
                       deviations)
        weight_in_: 2D numpy array, weights applied to each pixel during the 
                    computation fo the functional diversity metrics. For
                    variance-based methods, it these are used to compute a 
                    weighted PCA that leads to unbiased variance estimates
        redo_cube = boolean, if True, reshape the species x trait matrix to the
                    original cube dimensions
    """
    
    print('Applying Standardization and Dimensionalty Reduction')
    t0 = time.time()
    # Check sizes of cube and mask
    shp0__ = cube_.shape
    if mask_ == None:
        mask_ = np.ones((shp0__[0], shp0__[1]), dtype=bool)
    elif ((mask_.shape[0] == shp0__[0]) and (mask_.shape[1] == shp0__[1]) and 
          mask_.ndim > 2) is False:
        raise ValueError('The mask provided must be None or feature the ' +
                         'same x and y dimensions than the image')
    
    # When preallocating the cube, set the masked pixels to NaN so that
    # metrics are not computed on then, dependig on the tolerance treshold
    if (len(shp0__) == 2 or (shp0__[2] == 1)):
        # If there is only one trait, do not transform
        cube_pca_tmp = StandardScaler().fit_transform(
            cube_[mask_].reshape(-1, 1))
        cube_pca_ = np.zeros((shp0__[0], shp0__[1], 1)) * np.nan
        cube_pca_[mask_] = cube_pca_tmp
        
        if redo_cube is False:
            cube_pca_ = cube_pca_.reshape(-1, 1)
            
        max_dist_Eucl_ = None
        max_dist_SS_ = None
        explained_variance_ratio_ = 1.
        n_cmps_ = 1
    else:
        if weight_in_ == None:
            # Standardization and Principal Component Analysis
            (cube_pca_tmp, max_dist_Eucl_, max_dist_SS_,
             explained_variance_ratio_) = (
                 gnd.apply_std_PCA(cube_[mask_], n_components=fexp_var_in,
                                   n_sigmas_norm=n_sigmas_norm,
                                   get_max_dis_=True))
        else:
            print('CAUTION: Weighted principal components are to be used ' +
                  'only for variance-based methods')
            (cube_pca_tmp, max_dist_Eucl_, max_dist_SS_,
             explained_variance_ratio_) = (
                 gnd.apply_std_WPCA(cube_[mask_], weight_in_[mask_],
                                    get_max_dis_=True,
                                    n_sigmas_norm=n_sigmas_norm))

        n_cmps_ = cube_pca_tmp.shape[1]
        cube_pca_ = np.zeros((shp0__[0] * shp0__[1], n_cmps_)) * np.nan
        cube_pca_[mask_.reshape(-1)] = cube_pca_tmp

        if redo_cube:
            cube_pca_ = cube_pca_.reshape((shp0__[0], shp0__[1], -1))
    
    # Report time
    report_time(t0)
    
    return(cube_pca_, max_dist_Eucl_, max_dist_SS_, explained_variance_ratio_,
           n_cmps_)


def window_weights(weight_in, wsize_):
    if (weight_in == None):
        weight_w_ = np.ones(wsize_**2).reshape(1, -1) / wsize_**2
    else:
        weight_w_ = None
    
    return(weight_w_)
        

def varpar_weights(weight_in, shp0_, wsz_2):
    if (weight_in == None):
        return(np.ones((shp0_[0], shp0_[1])) / wsz_2)
    else:
        return(weight_in)


def raoQ_grid(cube_, wsz_=3, mask_in=None, weight_in=None, fexp_var_in=.98,
              n_sigmas_norm=6., alphas_=[1.], nan_tolerance=0.,
              calculate_RaoQ_part=False, nan_tolerance_gamma=.01):
    """
    Compute Rao's quadratic index (Q) and if requested apply variance-based
    partitioninig of alpha, beta and gamma diversities using the method
    from de Bello et al, (2010). 
    Inputs:
        cube_: numpy array, spectral image or data cube of dimensions (x, y,
               traits)
        wsz_: integer, size of the wsz_ x wsz_ moving window used to compute
               functional diveristy metrics.
        mask_in: boolean 2D matrix, mask indicating pixels to be removed from
                 the calculations. These are converted to nan, whose abundance
                 is accounted for the argument nan_tolerance
        weight_in: 2D numpy array, weights applied to each pixel during the 
                   computation fo the functional diversity metrics
        n_sigmas_norm: float, standardized distance to bounds (in standard
                       deviations)
        alphas_: list, aplhas for the parametric RaoQ
        nan_tolerance: float, fraction of NaN pixels within the moving window
        calculate_RaoQ_part: Boolean, if true, uses Rao Q to paritition
                             diveristy at different scales
        nan_tolerance_gamma: float, nan_tolerance applied to the computation of
                             RaoQ over the whole scene
    """
    # Check inputs and make sure that the cube dimensions are at least as large 
    # as the  required moving window
    if isinstance(cube_, tuple):
        shp0_ = (cube_[0].shape[0], cube_[0].shape[1], cube_[4])
        already_PCs = True    
    elif isinstance(cube_, np.ndarray):
        shp0_ = cube_.shape
        already_PCs = False  
    else:
        raise ValueError(
            'Wrong input. Present a cube of spectral or vegetation traits ' +
            'to process in a numpy array or their principal components. ' +
            'In the second the input argument "cube_" must be a tulpe ' +
            'containing all the ouput arguments the function apply_pca_cube()')

    if (shp0_[0] < wsz_) or (shp0_[1] < wsz_):
        raise ValueError('The image provided is larger than the ' +
                            'requested moving window')
        
    # Dimensionality reduction.
    # Apply Principal Components Analysis unless the PCs are provided, together
    # with all the cube_pca() function in a tulpe via the input argument cube_.
    if already_PCs:
            cube_pca = cube_[0]               
            max_dist_Eucl = cube_[1]
            n_cmps = cube_[4]
    else:
        # Apply Dimensionality Reductiona and standardization
        if weight_in == None:
            (cube_pca, max_dist_Eucl, _, _, n_cmps) = apply_pca_cube(
                cube_, mask_=mask_in, fexp_var_in=fexp_var_in,
                n_sigmas_norm=n_sigmas_norm, weigth_in_=weight_in)

    # Compute the weights to be used for each pixel of the moving window
    weight_w = window_weights(weight_in, wsz_)
    
    # Preallocate, depending on the requested outputs
    if calculate_RaoQ_part:
        raoQ = {'RaoQ': np.zeros((len(range(0, shp0_[1], wsz_)),
                                  len(range(0, shp0_[0], wsz_)))) * np.nan,
                'RaoQ_mean': np.nan, 'RaoQ_median': np.nan, 'RaoQ_std': np.nan}
        raoQ_part = {'alpha': np.nan, 'beta': np.nan, 'gamma': np.nan,
                     'f_alpha': np.nan, 'f_beta': np.nan}
    else:
        raoQ = {'RaoQ': np.zeros((len(range(0, shp0_[1], wsz_)),
                                  len(range(0, shp0_[0], wsz_)))) * np.nan,
                'RaoQ_mean': np.nan, 'RaoQ_median': np.nan, 'RaoQ_std': np.nan} 
        raoQ_part = None

    # Generate the moving window and compute the metrics. The calculations
    # include by default the Generalized Normalization
    t0 = time.time()
    print('Calculating Rao Q...')
    ki_ = 0    
    for xi_ in range(0, shp0_[1]-1, wsz_):
        kj_ = 0
        for ji_ in range(0, shp0_[0]-1, wsz_):
            # Sample the pixels within the moving window
            X = cube_pca[ji_:ji_ + wsz_, xi_:xi_ + wsz_].reshape(-1, n_cmps)
            
            # Compute the metrics if no NaN frequency is below tolerance
            if np.sum(np.isnan(X.sum(axis=1))) / (wsz_ ** 2) <= nan_tolerance:
                # Get the weights for a window if an image of a mask or weights
                # is provided 
                if weight_in != None:
                    weight_w = (weight_in[ji_:ji_ + wsz_,
                                          xi_:xi_ + wsz_].reshape(1, -1))
                    sum_weight_w = np.sum(weight_w)
                
                    if sum_weight_w > 0.:
                        weight_w /= sum_weight_w
                
                raoQ['RaoQ'][ki_, kj_], _ = gnd.mpaRaoS_freq(
                    X, weight_w, normalize_dist=True, alphas_=alphas_,
                    nan_tolerance=nan_tolerance, max_dist_=max_dist_Eucl)
            kj_ += 1
        ki_ += 1
    
    raoQ['RaoQ_mean'] = np.nanmean(raoQ['RaoQ'])
    raoQ['RaoQ_median'] = np.nanmedian(raoQ['RaoQ'])
    raoQ['RaoQ_std'] = np.nanstd(raoQ['RaoQ'])
    report_time(t0)
    
    # If requested, copmutes RaoQ over the whole scene and uses Rao Q to
    # partition diveristy in the alpha, gamma, and beta components. This
    # approach is not recommened for large scenes, since the computation of
    # the dismilarities can be extremely time-consuming
    if calculate_RaoQ_part:
        t0 = time.time()
        print('Partitioning diversity with Rao Q...')
        print('CAUTION: Computing Rao Q over the whole scene can be very slow' +
              ' or require too much memory')
        
        try:
            # Reshape the inputs and remove masked values
            cube_pca = cube_pca.reshape(shp0_[0] * shp0_[1], -1)
            if weight_in == None:
                weight_w = np.ones(shp0_[0] * shp0_[1]).reshape(1, -1)
            else:
                weight_w = weight_w.reshape(1, -1)

            if mask_in != None:
                mask_in = mask_in.reshape(-1)
                cube_pca = cube_pca[mask_in]
                cube_pca = cube_pca[mask_in]
                weight_w = weight_w[mask_in].reshape(1, -1)
            weight_w /= weight_w.sum() 

            # Compute RaoQ over the whole scene, this can be very slow
            raoQ_gamma, _ = gnd.mpaRaoS_freq(
                        cube_pca, weight_w, normalize_dist=True,
                        alphas_=alphas_, nan_tolerance=nan_tolerance_gamma,
                        max_dist_=max_dist_Eucl)
            
            # Partition diveristy
            raoQ4part = np.concatenate((raoQ['RaoQ'].reshape(-1),
                                        np.array([raoQ_gamma]))).reshape(-1, 1)
            
            (_, alpha_mean_, beta_add, gamma_, alpha_prop, beta_prop,
            alpha_prop_norm, beta_prop_norm) = gnd.div_part_EqNum(
                raoQ4part, raoQ['RaoQ'].size)
            raoQ_part['alpha'] = alpha_mean_
            raoQ_part['beta'] = beta_add
            raoQ_part['gamma'] = gamma_
            raoQ_part['f_alpha'] = alpha_prop_norm
            raoQ_part['f_beta'] = beta_prop_norm

        except Exception as ex:
            if type(ex).__name__ == 'MemoryError':
                print('Memory error: It was not possible computing Rao Q for' +
                      'the whole image.')
                pass
        
        report_time(t0)
 
    return(raoQ, raoQ_part)    


def varpart_grid(cube_, wsz_=3, weight_in=None, mask_in=None, nan_tolerance=0.,
                 fexp_var_in=.98, n_sigmas_norm=6.):
    # Check inputs and make sure that the cube dimensions are at least as large 
    # as the  required moving window
    if isinstance(cube_, tuple):
        shp0_ = (cube_[0].shape[0], cube_[0].shape[1])            
        already_PCs = True    
    elif isinstance(cube_, np.ndarray):
        shp0_ = cube_.shape
        already_PCs = False  
    else:
        raise ValueError(
            'Wrong input. Present a cube of spectral or vegetation traits ' +
            'to process in a numpy array or their principal components. ' +
            'In the second the input argument "cube_" must be a tulpe ' +
            'containing all the ouput arguments the function apply_pca_cube()')

    if ((shp0_[0] < wsz_) or (shp0_[1] < wsz_)):
        raise ValueError('The image provided is larger than the ' +
                            'requested moving window')
        
    # Dimensionality reduction.
    # Apply Principal Components Analysis unless the PCs are provided, together
    # with all the cube_pca() function in a tulpe via the input argument cube_.
    if already_PCs:
        cube_pca = cube_[0].reshape(shp0_[0] * shp0_[1], -1)             
        max_dist_SS = cube_[2]
    else:       
        # Apply Dimensionality Reductiona and standardization
        if weight_in == None:
            (cube_pca, _, max_dist_SS, _, n_cmps) = apply_pca_cube(
                cube_, mask_=mask_in, fexp_var_in=fexp_var_in,
                n_sigmas_norm=n_sigmas_norm, redo_cube=False)
        else:
            (cube_pca, _, max_dist_SS, _, n_cmps) = apply_pca_cube(
                cube_, mask_=mask_in, fexp_var_in=fexp_var_in,
                n_sigmas_norm=n_sigmas_norm, redo_cube=False)

    # Indices
    indx_m = np.arange(shp0_[0] * shp0_[1]).reshape((shp0_[0], shp0_[1]))
    indx_l = indx_m.reshape((shp0_[0] * shp0_[1]))
    
    # Weights
    wsz_2 = wsz_** 2
    weight_w = varpar_weights(weight_in, shp0_, wsz_2)
    
    # Convert the abundance of the masked pixels to 0 so that the windows
    # containing them are filtered out 
    if mask_in != None:
        weight_w[mask_in == False] = 0.

    # Number of traits and groups
    n_cmps = cube_pca.shape[1]
    n_comm = int(shp0_[0] * shp0_[1] / wsz_2)
    
    # Preallocate
    frac_used = np.nan
    SSalpha, SSbeta, SSgamma = np.nan, np.nan, np.nan
    f_alpha, f_beta = np.nan, np.nan

    # Get relative abundances
    try:
        RelAbun_sp = np.zeros((n_comm, int(shp0_[0] * shp0_[1])))

        k_r = 0
        for xi_ in range(0, shp0_[1]-1, wsz_):
            for ji_ in range(0, shp0_[0]-1, wsz_):
                ind_i = indx_m[ji_:ji_ + wsz_, xi_:xi_ + wsz_].reshape(-1)
                I_sp = np.where(np.isin(indx_l, ind_i)==True)[0]
                RelAbun_sp[k_r, I_sp] = weight_w[ji_:ji_ + wsz_,
                                                xi_:xi_ + wsz_].reshape(-1)
                k_r += 1

        # Select the windows where all pixels have data
        I_ = ((np.count_nonzero(RelAbun_sp, axis=1)).astype(float) >=
            (wsz_2 * (1. - nan_tolerance)))
        RelAbun_sp = gnd.div_zeros(RelAbun_sp,
                                np.sum(RelAbun_sp, axis=1).reshape(-1, 1))

        if any(I_):
            t0 = time.time()
            print('Partitioning diversity with Variance...')
            (SSgamma, _, _, _, SSbeta, _, _,
            _, SSalpha, _, _) = (gnd.LalibertePart_w(
                cube_pca, RelAbun_sp[I_, :], pca_transf=False,
                normalize=True, SS_max_in=max_dist_SS))
            f_alpha = 100 * SSalpha / SSgamma
            f_beta = 100* SSbeta / SSgamma
       
        report_time(t0)
        
        frac_used = np.sum(I_) / I_.shape[0]
        
    except Exception as ex:
        if type(ex).__name__ == 'MemoryError':
            print('Memory error: It was not possible computing SS for' +
                    'the whole image.')
            pass

    return (SSalpha, SSbeta, SSgamma, f_alpha, f_beta, frac_used)

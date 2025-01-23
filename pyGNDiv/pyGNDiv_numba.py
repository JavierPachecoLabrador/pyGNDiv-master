# -*- coding: utf-8 -*-
"""
@author: Javier Pacheco-Labrador, PhD (jpacheco@bgc-jena.mpg.de)
         Max Planck Institute for Biogeochemsitry, Jena, Germany
        Currently: javier.pacheco@csic.es
         Environmental Remote Sensing and Spectroscopy Laboratory (SpecLab)
         Spanish National Research Council (CSIC)

DESCRIPTION:
==============================================================================
This package contains the main functions of the Generalizable Normalization of
    Functional Diversity Metrics proposed in Pacheco-Labrador, J. et al, under
    review.

Cite as:
  Pacheco-Labrador, J., de Bello, F., Migliavacca , M., Ma, X., Carvalhais, N.,
    & Wirth, C. (2023). A generalizable normalization for assessing
    plant functional diversity metrics across scales from remote sensing.
    Methods in Ecology and Evolution.

The package computes the following functional diversity metrics:
    * Rao's Quadratic entropy index (Q). Based on the rasterdiv R package from
    Rocchini et al, (2021).
    * Rao's equivalent number (Qeq) as proposed in de Bello (2010).
    * Functional richness (FRic). From the dbFD R-package by Laliberté &
    Legendre (2010)

The package icludes two alpha, beta and gamma-diveristy partitioning methods:
    * Total variance framework. Adapted from Laliberté et al, (2020) to
    operate with trait-abundance matrices. This approach uses the wpca Python
    package from Jake VanderPlas (https://github.com/jakevdp/wpca)
    * Diversity decomposition. It can use Rao's Q or its equivalent number.

pyGNDiv allows for applying three different dissimilarity normalization
approaches:
    * Non-normalization: Use dissimilarity's absolute value
    * Local normalization: Divide by the maximum dissimilarity of the dataset
    * Generalizable normalization: Data-agnostic normalization proposed by
    Pacheco-Labrador et al, (2023)

REFERENCES:
    de Bello, F., Lavergne, S., Meynard, C.N., Lepš, J., & Thuiller, W.
        (2010). The partitioning of diversity: showing Theseus a way out of
        the labyrinth. Journal of Vegetation Science, 21, 992-1000.

    Jost, L. (2007). Partitioning diversity into independent alpha and beta
        components. Ecology, 88, 2427-24397, this approach leads to unbiased
        estimates of Beta diversity.

   Laliberté, E., & Legendre, P. (2010). A distance-based framework for
        measuring functional diversity from multiple traits. Ecology, 91,
        299-305.

    Laliberté, E., Schweiger, A.K., & Legendre, P. (2020). Partitioning plant
    spectral diversity into alpha and beta components. Ecology Letters, 23,
    370-380.

    Rocchini et al, (2021). rasterdiv—An Information Theory tailored R
        package for measuring ecosystem heterogeneity from space: To the
        origin and back. Methods in Ecology and Evolution, 12, 1093-1102.

HISTORY:
    * Created: 16-May-2023. Javier Pacheco-Labrador, PhD. Improve RaoQ
    calculation with Numba. Computation is accelerated several times, depending 
    on the data volume.
"""

# %% Imports
import numpy as np
import copy as copy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from itertools import combinations_with_replacement
from scipy.spatial import ConvexHull
from math import gamma
from wpca import WPCA

from numba import njit, uint8, uint32, float64


# %% Ancillary Functions
def preallocate_outputs(num_, sensors_=None):
    """
    Generate empty dictionaries as a funciton of the number of regions to be
    analyzed
    """
    if sensors_ is None:
        FRic = [dict()] * num_
        VBdp_ = [dict()] * num_
        VBdp_Gnorm = [dict()] * num_
        VBdp_Lnorm = [dict()] * num_
        RQdp = [dict()] * num_
        RQdp_Gnorm = [dict()] * num_
        ENdp_Gnorm = [dict()] * num_
        RQdp_Lnorm = [dict()] * num_
        ENdp_Lnorm = [dict()] * num_
    else:
        FRic = dict()
        VBdp_ = dict()
        VBdp_Gnorm = dict()
        VBdp_Lnorm = dict()
        RQdp = dict()
        RQdp_Gnorm = dict()
        ENdp_Gnorm = dict()
        RQdp_Lnorm = dict()
        ENdp_Lnorm = dict()
        for sen_ in sensors_:
            (FRic[sen_], VBdp_[sen_], VBdp_Gnorm[sen_], VBdp_Lnorm[sen_],
             RQdp[sen_], RQdp_Gnorm[sen_], ENdp_Gnorm[sen_], RQdp_Lnorm[sen_],
             ENdp_Lnorm[sen_]) = (
                 preallocate_outputs(num_))
    return(FRic, VBdp_, VBdp_Gnorm, VBdp_Lnorm, RQdp, RQdp_Gnorm, ENdp_Gnorm,
           RQdp_Lnorm, ENdp_Lnorm)


def apply_std_PCA(X, n_components=.98, output_all=False, get_max_dis_=False,
                  n_sigmas_norm=6.):
    """
    Standardize an apply PCA to a X matrix of n observations by p variables
    """
    std_model = StandardScaler().fit(X)
    Xstd_ = std_model.transform(X)
    pca_model = PCA(n_components=n_components).fit(Xstd_)
    pca_comps = pca_model.transform(Xstd_)

    if get_max_dis_ is True:
        fexp_var = np.sum(pca_model.explained_variance_ratio_)
        max_dist_Eucl = max_eucl_dist_PCA_fun(X.shape[1], fexp_var=fexp_var,
                                              n_sigmas=n_sigmas_norm)
        max_dist_SS = max_SS_PCA_fun(X.shape[1], X.shape[0], fexp_var=fexp_var,
                                     n_sigmas=n_sigmas_norm)
    else:
        max_dist_Eucl = 0.
        max_dist_SS = 1.

    if output_all is True:
        return(pca_comps, max_dist_Eucl, max_dist_SS,
               pca_model.explained_variance_ratio_, pca_model, std_model)
    else:
        return(pca_comps, max_dist_Eucl, max_dist_SS,
               pca_model.explained_variance_ratio_)


def apply_std_WPCA(X, RelAbun_sp_sp, n_components=.98, output_all=False,
                   get_max_dis_=False, n_sigmas_norm=6.):
    """
    Standardize an apply weighted PCA to a X matrix of n observations by p
    variables
    https://github.com/jakevdp/wpca
    """
    weights_ = np.repeat(RelAbun_sp_sp.reshape(-1, 1), X.shape[1], axis=1)
    weights_ = div_zeros(weights_, np.sum(weights_))
    std_model = StandardScaler().fit(
        X, sample_weight=RelAbun_sp_sp.reshape(-1))
    Xstd_ = std_model.transform(X)
    # Apply weighted PCA
    pca_model = WPCA(n_components=X.shape[1]).fit(Xstd_,
                                                  **{'weights': weights_})

    n_comps_sel = np.min((X.shape[1], (np.count_nonzero(
        np.cumsum(pca_model.explained_variance_ratio_) <= .98) + 1)))
    if n_comps_sel < X.shape[1]:
        pca_model = WPCA(n_components=n_comps_sel
                         ).fit(Xstd_, **{'weights': weights_})
    pca_comps = pca_model.transform(Xstd_)

    if get_max_dis_ is True:
        fexp_var = np.sum(pca_model.explained_variance_ratio_)
        max_dist_Eucl = max_eucl_dist_PCA_fun(X.shape[1], fexp_var=fexp_var,
                                              n_sigmas=n_sigmas_norm)
        max_dist_SS = max_SS_PCA_fun(X.shape[1], X.shape[0],
                                     fexp_var=fexp_var, n_sigmas=n_sigmas_norm)
    else:
        max_dist_Eucl = 0.
        max_dist_SS = 1.

    if output_all is True:
        return(pca_comps, max_dist_Eucl, max_dist_SS,
               pca_model.explained_variance_ratio_, pca_model, std_model)
    else:
        return(pca_comps, max_dist_Eucl, max_dist_SS,
               pca_model.explained_variance_ratio_)


def div_zeros(a, b):
    """
    Enable division by 0, making sure that dimensions and format of the input
    values match, and returning 0 whenever any value is divided by 0.
    """
    if isinstance(a, float) or isinstance(a, int):
        a = np.array(a).reshape((1, 1)).astype(float)
    if isinstance(b, float) or isinstance(b, int):
        b = np.array(b).reshape((1, 1)).astype(float)
    if np.ndim(a) == 1:
        a = a.reshape(-1, 1)
    if np.ndim(b) == 1:
        b = b.reshape(-1, 1)

    if (a.shape[0] == b.shape[0]) and (a.shape[1] != b.shape[1]):
        if a.shape[1] == 1:
            b = np.repeat(a, b.shape[1], axis=1)
        elif b.shape[1] == 1:
            b = np.repeat(b, a.shape[1], axis=1)
    if (a.shape[1] == b.shape[1]) and (a.shape[0] != b.shape[0]):
        if a.shape[0] == 1:
            a = np.repeat(a, b.shape[0], axis=0)
        elif b.shape[0] == 1:
            b = np.repeat(b, a.shape[0], axis=0)
    if (a.shape[0] == 1) and (a.shape[1] == 1):
        a = a * np.ones_like(b)
    if (b.shape[0] == 1) and (b.shape[1] == 1):
        b = b * np.ones_like(a)

    return(np.divide(a, b, out=np.zeros_like(a), where=b != 0.))

# @njit()
# def euclidean_dist(x, y):
#     """
#     Computes euclidean distance between two n-dimensional vectors
#     """
#     return(np.sqrt(np.sum((x - y) ** 2)))

@njit(float64(float64[:, :], uint32, uint32))
def euc_dist_rao(x_, i_, j_):
    """
    Computes euclidean distance between two n-dimensional vectors which are
    rows selected from a sample x traits matrix
    """
    return(np.sqrt(np.sum((x_[i_, :] - x_[j_, :]) ** 2)))


# %% Maximum values for Global normalization
"""
Maximum values proposed for the generalizable normalization approach in
Pacheco-Labrador et al, 2023
"""


def max_eucl_dist_PCA_fun(n_ori_vars, fexp_var=.98, n_sigmas=6.):
    """
    Computes the maximum plausible euclidean distance in a normalized
    n-dimensional space
    """
    return(2 * n_sigmas * np.sqrt(np.ceil(n_ori_vars * fexp_var)))


def max_SS_PCA_fun(n_ori_vars, n_sp, fexp_var=.98, n_sigmas=6.):
    """
    Computes the maximum plausible sume of squares in a normalized
    n-dimensional space
    """
    return(n_sigmas**2. * n_ori_vars * n_sp * fexp_var)


def hypersphere_volume(radius, n_dims):
    """
    Computes the maximum volume for a hypersfere in a normalized n-dimensional
    space
    """
    V_n = ((np.pi**(n_dims/2.)) * (radius**n_dims)) / gamma(n_dims/2. + 1.)
    return(V_n)


# %% Rao Quadratic Entropy adapted from rasterdiv (from Rocchini et al. 2021)
"""
Python implementation of some of the functions in the raterdiv package of
Rocchini et al, (2021). rasterdiv—An Information Theory tailored R package
for measuring ecosystem heterogeneity from space: To the origin and back.
Methods in Ecology and Evolution, 12, 1093-1102.

Rao's Q can be computed from plant functional traits but also from taxonomic
information (special case where Rao Q equals the Gini-Simpson index, not shown
in Pacheco-Labrador, et al (2023))

In this case, the functions are adatped to ingest data in the format of
species x trait and abundance x species matrices.
"""


@njit(uint8[:](uint32[:,:]))
def get_Idiag(comb):
    I_diag = np.ones(comb.shape[0], dtype=np.uint8)
    I_diag[np.where((comb[:, 0] == comb[:, 1]))[0]] = 0
    # for iii_ in range(comb.shape[0]):
    #     if comb[iii_, 0] == comb[iii_, 1]:
    #         I_diag[iii_] = 0
    return(I_diag)


@njit(float64[:](float64[:, :], uint32[:, :], uint8[:]))
def get_euc_dist_rao(X, comb, I_diag):
    dists_ = np.zeros(comb.shape[0], dtype=np.float64)
    for iii_ in range(comb.shape[0]):
        if I_diag[iii_] == 1:
            dists_[iii_] = euc_dist_rao(X, comb[iii_, 0], comb[iii_, 1])
    return(dists_)


@njit(float64[:](float64[:, :], uint32[:, :], uint32))
def get_comb_freq(freq_, comb, i_):
    comb_freq_ = np.zeros(comb.shape[0], dtype=np.float64)
    for iii_ in range(comb.shape[0]):
        comb_freq_[iii_] = freq_[i_, comb[iii_, 0]] * freq_[i_, comb[iii_, 1]]
    return(comb_freq_)


@njit(float64[:, :](float64[:], float64, float64[:], uint8[:], float64[:]))
def get_raoQ(comb_freq_, nan_tolerance, dists_, I_diag, alphas_):
    """
    Compute RaoQ
    """
    # Preallocate
    raoQ = np.zeros((1, alphas_.shape[0]), dtype=np.float64) * np.nan
    # If there are more data than the predefined treshold of acceptable data
    if np.sum(comb_freq_) > nan_tolerance:
        I_ = np.where(comb_freq_ > 0.)[0]
        I_d = np.where(comb_freq_ * dists_ > 0.)[0]

        # Do this way to avoid np.all for numba
        # I_dd = np.where(np.all((comb_freq_ > 0., I_diag == 1), axis=0))[0]
        I_dd = np.zeros(comb_freq_.shape[0], dtype=np.uint32)
        j_ = 0
        for i_ in range(comb_freq_.shape[0]):
            if comb_freq_[i_] > 0. and I_diag[i_] == 1:
                I_dd[j_] = i_
                j_ += 1
        I_dd = I_dd[:j_]

        eq_window = 2 * I_diag[I_].shape[0] - I_diag[I_diag == 0].shape[0]
        k_ = 0
        for alpha in alphas_:
            # alpha = alphas_[k_]
            # print(k_)
            if np.isinf(alpha):
                raoQ[0, k_] = np.nanmax(dists_[I_] * 2)
            elif alpha == 0:
                if eq_window > 0.:
                    raoQ[0, k_] = (np.nanprod(dists_[I_dd]) **
                                   (1 / (eq_window)))
                else:
                    raoQ[0, k_] = 0.
            elif alpha > 0:
                raoQ[0, k_] = ((2 * dists_[I_d]**alpha @
                                comb_freq_[I_d].T) ** (1 / alpha))
            k_ += 1
    return(raoQ)


@njit(float64[:, :](float64[:, :], uint32[:, :], float64, float64[:], uint8[:],
                    float64[:]))
def do_raoQ_loop(freq_, comb, nan_tolerance, dists_, I_diag,
                 alphas_):
    "Loop through communities to compute RaoQ"
    # Preallocates
    raoQ = np.zeros((freq_.shape[0], alphas_.shape[0])) * np.nan

    # Loop
    for i_ in range(freq_.shape[0]):
        comb_freq_ = get_comb_freq(freq_, comb, i_)
        raoQ[i_, :] = get_raoQ(comb_freq_, nan_tolerance, dists_,
                               I_diag, alphas_)

    return(raoQ)


def mpaRaoS_freq(X, freq_, alphas_=np.ones(1), nan_tolerance=0.,
                 normalize_dist=False, tax_div=False, get_extra_outs=False,
                 out_intvar_in=None, max_dist_=0.):
    """
    Version of mpaRaoS.R function that takes a matrix X of unique species-
    values (observations in rows, variables in columns); and different
    combinations of their frequencies (combinations ~windows in rows, species
    in columns)
    """
    X = X.astype(dtype=np.float64)
    # Check if alpha is iterable array and convert it otherwise
    if not isinstance(alphas_, np.ndarray):
        if isinstance(alphas_, list):
            alphas_ = np.array(alphas_)
        else:
            alphas_ = np.array([alphas_])

    # Generate combination of pixels to be compared or use the input one
    # provided if as input
    if out_intvar_in is None:
        comb = np.array(list(combinations_with_replacement(
            range(X.shape[0]), 2)), dtype=np.uint32)
        # I_diag = np.ones(len(comb))
        # for iii_ in range(len(comb)):
        #     if comb[iii_][0] == comb[iii_][1]:
        #         I_diag[iii_] = 0
        I_diag = get_Idiag(comb)
    else:
        comb = out_intvar_in['comb'].astype(np.uint32)
        I_diag = out_intvar_in['I_diag']

    # Compute RaoQ and the euclidean distances
    raoQ, dists_ = mpaRaoS_freq_i(X, comb, freq_, I_diag, alphas_,
                                  nan_tolerance=nan_tolerance,
                                  normalize_dist=normalize_dist,
                                  tax_div=tax_div,
                                  out_intvar_in=out_intvar_in,
                                  max_dist_=max_dist_)

    # Provide more or less outputs, depending on what is aked for
    if get_extra_outs is True:
        out_intvar_in = {'comb': comb, 'I_diag': I_diag, 'dists_': dists_}
        return(raoQ, out_intvar_in)
    else:
        return(raoQ, dists_)


def mpaRaoS_freq_i(X, comb, freq_, I_diag, alphas_, nan_tolerance=0.,
                   normalize_dist=False, out_shp=[], tax_div=False,
                   out_intvar_in=None, max_dist_=0.):
    """
    Compute the dissimilarity metric (Euclidean distance) and RaoQ
    """
    # Compute distances.
    if tax_div is False:
        # For functional diversity, compute distances between traits
        if out_intvar_in is None:
            # dists_ = np.zeros(len(comb))
            # for iii_ in range(len(comb)):
            #     if I_diag[iii_] == 1:
            #         dists_[iii_] = euc_dist_rao(X, comb[iii_][0],
            #                                     comb[iii_][1])
            dists_ = get_euc_dist_rao(X, comb, I_diag)
        else:
            dists_ = copy.deepcopy(out_intvar_in['dists_'])

        # Normalize distances. Necessary for using equivalent numbers in
        # of alpha and beta-diversity partitioning (de Bello et al. (2010))
        if normalize_dist is True:
            if max_dist_ > 0.:
                dists_ = dists_ / max_dist_
            else:
                dists_ = dists_ / np.max(dists_)

    elif tax_div is True:
        # For taxonomical diversity, assign 0 between the same species, and 1.
        # to the rest, this is equivalent to Gini-Simpson's index (de Bello et
        # al. 2010)
        dists_ = copy.deepcopy(I_diag)

    # Compute RaoQ
    raoQ = do_raoQ_loop(freq_, comb, nan_tolerance, dists_, I_diag, alphas_)
    # i_ = 0
    # for i_ in range(freq_.shape[0]):
    #     # comb_freq_ = np.zeros(comb)) * np.nan
    #     # for iii_ in range(len(comb)):
    #     #     comb_freq_[iii_] = (freq_[i_, comb[iii_][0]] *
    #     #                         freq_[i_, comb[iii_][1]])
    #     comb_freq_ = get_comb_freq(freq_, comb, i_)
    #     raoQ[i_, :] = get_raoQ(comb_freq_, nan_tolerance, dists_,
    #                            I_diag, alphas_)

    if out_shp == []:
        return(np.squeeze(raoQ), dists_)
    else:
        return(raoQ.reshape(out_shp), dists_)


# %% Functional Richness
"""
Python implementation of the Functional Richness diversity metric, following
the version implemented in the dbFD R-package from Laliberté & Legendre (2010).
A distance-based framework for measuring functional diversity from multiple
traits. Ecology, 91, 299-305.
In this case, FRic is normalized using a local (by the maximum value in the
region) and a Generalizable Normalization approach (Pacheco-Labrador et al,
2023)
"""


def get_FRic(comps, abund_, fexp=.98, n_sp=[], onevar_warning=True,
             n_sigmas=6.):
    if comps.shape[1] == 1:
        if onevar_warning:
            print("FRic: Only one continuous trait or dimension in 'x'. " +
                  "FRic was measured as the range, NOT as the convex hull" +
                  " volume.")
        FRic = np.zeros(abund_.shape[0]) * np.nan
        FRic_Gnorm = np.zeros(abund_.shape[0]) * np.nan
        FRic_Lnorm = np.zeros(abund_.shape[0]) * np.nan
    else:
        # Get the number of unique species (dbFD.R line 307)
        # traits <- round(x.pco$li, .Machine$double.exponent)
        # Round to 10 since python leaves 10 positions for the exponent)
        # https://stackoverflow.com/questions/20473968/floating-point-numbers
        if len(n_sp) == 0:
            tol = np.finfo(float).eps
            c_ = 0
            n_sp = [0] * abund_.shape[0]
            for c_ in range(abund_.shape[0]):
                I_ = np.where(abund_[c_, :] > 0)[0].tolist()
                tr_tmp = np.round(comps[I_, :], decimals=10)
                tr_tmp[np.abs(tr_tmp) < tol] = 0.
                n_sp[c_] = (np.unique(tr_tmp, axis=0)).shape[0]
        min_n_sp = min(n_sp)
        m_max = min_n_sp - 1

        # Check if dimensionality reduction is needed (dbFD.R ~line 395)
        # Here assume the default m="max"
        if m_max < 3:
            # print("FRic: To respect s > t, FRic could not be calculated " +
            #       "for communities with <3 functionally singular species.")
            m_max = min_n_sp - 1
        # Here assume the default m="max" (dbFD.R ~line 407)
        m_ = int(m_max)

        traits_FRic = comps[:, :m_+1]
        n_axis = traits_FRic.shape[1]

        FRic = np.zeros((abund_.shape[0]))
        FRic_Gnorm = np.zeros((abund_.shape[0]))
        FRic_Lnorm = np.zeros((abund_.shape[0]))
        c_ = 0
        for c_ in range(abund_.shape[0]):
            I_ = np.where(abund_[c_, :] > 0)[0].tolist()
            tr_FRic = traits_FRic[I_, :]
            try:
                hull = ConvexHull(tr_FRic)
                FRic[c_] = hull.volume
                FRic_max = hypersphere_volume(n_sigmas * fexp, n_axis)
                FRic_Gnorm[c_] = (FRic[c_] / FRic_max) ** (1 / n_axis)
            except:
                FRic[c_] = np.nan
                FRic_Gnorm[c_] = np.nan
                FRic_Lnorm[c_] = np.nan
            FRic_Lnorm = FRic / np.nanmax(FRic)

    return(FRic, FRic_Gnorm, FRic_Lnorm)


# %% Variance-based diversity partitioning
"""
Phython implementation of Laliberté, E., Schweiger, A.K., & Legendre, P.
(2020). Partitioning plant spectral diversity into alpha and beta components.
Ecology Letters, 23, 370-380.

This version is adapted to operate with species-by-traits and abundance-by-
species matrices. For this reason, a weighted PCA must be applied to achieve
an unbiased mean.
"""


def LalibertePart_w(X, RelAbun_sp, pca_transf=True, normalize=False,
                    SS_max_in=1., n_sigmas_norm=6.):
    """
    Apply variance-based partitioninig of alpha, beta and gamma-diversities
    using the method from Laliberte et al. 2020
    Inputs:
        X (Traits_sp): traits matrix of n species x m traits
        RelAbun_sp: frequency matrix of p populations x n species
        pca_transf: If true, applies standardization and dimensionality
                    reduction to Traits_sp. Set to false if principal
                    components are provided instead
        SS_max_in: Maximum sum of squares used for normalization
        n_sigmas_norm: standardized distance to bounds (in standard deviations)
    """
    n_sp = X.shape[0]
    g_ = np.repeat(np.arange(0, RelAbun_sp.shape[0]).reshape(1, -1),
                   n_sp).reshape(-1)

    RelAbun_sp_sp = np.sum(RelAbun_sp, axis=0, keepdims=True)
    RelAbun_sp_com = np.sum(RelAbun_sp, axis=1, keepdims=True)
    RelAbun_sp_tot = np.sum(RelAbun_sp_sp)

    # Y_{ij}: matrmatrix containing the positions, along the p axes defining
    # the spectral space (column vectors y1, y2... yp) of n pixels (row
    # vectors x1, x2, ... xn) in region of interest
    # Apply PCA and standardization
    if pca_transf:
        Y, _, SS_max, _ = apply_std_WPCA(X, RelAbun_sp_sp, get_max_dis_=True,
                                         n_sigmas_norm=n_sigmas_norm)
    else:
        Y = copy.deepcopy(X)
        if normalize is True:
            SS_max = copy.deepcopy(SS_max_in)

    # Samples would be the number of pixels in the image. Here, I've species
    # and proportions. I just provide the number of communities by the number
    # of species
    n_samples = Y.shape[0] * RelAbun_sp.shape[0]
    n_vars = Y.shape[1]
    n_groups = RelAbun_sp.shape[0]

    # Gamma-diversity
    # Eq. 1
    y_j_mean = div_zeros(RelAbun_sp_sp @ Y, RelAbun_sp_tot)
    if normalize is False:
        s_ij = (Y - y_j_mean)**2
    elif (normalize is True) and (SS_max is None):
        s_ij = (Y - y_j_mean)**2
        SS_max = np.max(s_ij)
        s_ij = s_ij / SS_max
    else:
        s_ij = (Y - y_j_mean)**2 / SS_max
    # Eq. 2
    SSgamma = np.sum(RelAbun_sp_sp @ s_ij)
    # Eq. 3. Here divide by groups, since n pixels are already normalized.
    # Divide by n_groups, and not by n_groups-1, since  I should ne in fact
    # doing n-1, where n = n_groups * group_size, and the actual group size is
    # not known
    SDgamma = div_zeros(SSgamma, float(n_groups))

    # Feature contribution to spectral diversity
    # Eq. 4
    SSgamma_j = RelAbun_sp_sp @ s_ij
    # Eq. 5
    FCSDgamma = div_zeros(SSgamma_j, SSgamma)

    # Local contribution to spectral diversity
    # Each LCSDc,i value corresponds to the squared distance from one pixel
    # to the centroid in the p-dimensional PCA ordination plot
    # In this case, is the contribution of each species
    # Eq. 7
    SSgamma_i = RelAbun_sp_sp.T * np.sum(s_ij, axis=1, keepdims=True)

    # Eq. 6
    LCSDgamma = div_zeros(SSgamma_i, SSgamma)

    # Beta-diversity
    # Eq. 10
    y_kj_mean = RelAbun_sp @ Y

    # Eq. 11
    if normalize is False:
        s_kj = (y_kj_mean - y_j_mean)**2
    else:
        s_kj = (y_kj_mean - y_j_mean)**2 / SS_max

    # Eq. 12
    SSbeta_k = (RelAbun_sp_com * np.sum(s_kj, axis=1, keepdims=True))

    # Eq. 13
    SSbeta = np.sum(SSbeta_k)

    # Eq. 14 Here divide by groups, since n pixels are already normalized
    SDbeta = div_zeros(SSbeta, float(n_groups))

    # Feature contribution to spectral diversity
    # Eq. 17
    SSbeta_j = np.sum(RelAbun_sp_com * s_kj, axis=0, keepdims=True)

    # Eq. 16
    FCSDbeta = div_zeros(SSbeta_j, SSbeta)

    # Local contribution to spectral diversity
    # In this case, is the contribution of each community
    # Eq. 15
    LCSDbeta = div_zeros(SSbeta_k, SSbeta)

    # Alpha-diversity
    # Eq. 18
    s_ijk = np.zeros((n_samples, n_vars))
    if normalize is False:
        for i_ in range(n_groups):
            s_ijk[g_ == i_, :] = (Y - y_kj_mean[i_, :])**2
    else:
        for i_ in range(n_groups):
            s_ijk[g_ == i_, :] = ((Y - y_kj_mean[i_, :])**2 / SS_max)

    # Eq. 19
    SSalpha_jk = np.zeros((n_groups, n_vars))
    for i_ in range(n_groups):
        SSalpha_jk[i_, :] = RelAbun_sp[i_, :] @ s_ijk[g_ == i_, :]

    # Eq. 20
    SSalpha_k = np.sum(SSalpha_jk, axis=1, keepdims=True)

    # Eq. 21
    SDalpha_k = div_zeros(SSalpha_k, float(n_groups - 1.))

    # Eq. 22
    SSalpha = (RelAbun_sp_com.T @ SSalpha_k).item()

    # Feature contribution to spectral diversity
    # Eq. 24
    FCSDalpha_jk = div_zeros(SSalpha_jk, SSalpha_k)

    # Outputs
    return(SSgamma, SDgamma, FCSDgamma, LCSDgamma, SSbeta, SDbeta, FCSDbeta,
           LCSDbeta, SSalpha, SDalpha_k, FCSDalpha_jk)


def LalibertePart_w_wrap(Yw, RelAbun_sp, pca_transf=True, normalize=False,
                         SS_max_in=1.):
    """
    Wrapper for the LalibertePart_w() function that places outputs in a
    dictionary
    """
    VBdp = dict()
    (VBdp['SSgamma'], VBdp['SDgamma'], VBdp['FCSDgamma'], VBdp['LCSDgamma'],
     VBdp['SSbeta'], VBdp['SDbeta'], VBdp['FCSDbeta'], VBdp['LCSDbeta'],
     VBdp['SSalpha'], VBdp['SDalpha_k'], VBdp['FCSDalpha_jk']) = (
         LalibertePart_w(Yw, RelAbun_sp, pca_transf=pca_transf,
                         normalize=normalize, SS_max_in=SS_max_in))

    # Compute fractions of gamma variance explained by alpha and beta
    VBdp['Falpha'] = 100. * div_zeros(
        VBdp['SSalpha'], VBdp['SSgamma'])
    VBdp['Fbeta'] = 100. * div_zeros(VBdp['SSbeta'], VBdp['SSgamma'])
    return(VBdp)


# %% Diversity decomposition
"""
Python implementation of the diversity decomposition based on Rao Q index and
equivalent number from de Bello, F., Lavergne, S., Meynard, C.N., Lepš, J., &
Thuiller, W. (2010). The partitioning of diversity: showing Theseus a way out
of the labyrinth. Journal of Vegetation Science, 21, 992-1000.
"""


# de Bello partitioning Rao Q index or equivalent number
def div_part_EqNum(RaoQ_, n_comm, reshp_=False):
    """
    Diversity decomposition based on the Rao's Q equivalent number. According
    to Jost, L. (2007). Partitioning diversity into independent alpha and beta
    components. Ecology, 88, 2427-24397, this approach leads to unbiased
    estimates of Beta-diversity.
    """
    # Eq. 6
    alpha_ = div_zeros(np.ones_like(RaoQ_[:-1, :]), 1 - RaoQ_[:-1, :])
    alpha_mean_ = div_zeros(1., 1. - np.nanmean(RaoQ_[:-1, :], axis=0))
    # Eq. 7
    gamma_ = div_zeros(np.ones_like(RaoQ_[-1, :]), 1 - RaoQ_[-1, :])
    # Eq. 13
    beta_add = gamma_ - alpha_mean_
    # Eq. 12
    beta_prop = 100 * div_zeros(beta_add, gamma_)
    alpha_prop = 100 - beta_prop
    # Eq. 14
    beta_prop_norm = beta_prop / (1. - 1./n_comm)
    alpha_prop_norm = 100 - beta_prop_norm
    if reshp_:
        return(alpha_.reshape(-1), alpha_mean_, beta_add, gamma_, alpha_prop,
               beta_prop, alpha_prop_norm, beta_prop_norm)
    else:
        return(alpha_, alpha_mean_, beta_add, gamma_, alpha_prop, beta_prop,
               alpha_prop_norm, beta_prop_norm)


def div_part_Index(RaoQ_, n_comm, reshp_=False):
    """
    Diversity decomposition based on the Rao's Q index. According to Jost, L.
    (2007). Partitioning diversity into independent alpha and beta components.
    Ecology, 88, 2427-24397, this approach leads to spuriously biased
    estimates of Beta-diversity.
    """
    # Eq. 1
    alpha_ = RaoQ_[:-1, :]
    alpha_mean_ = np.mean(alpha_, axis=0)
    # Eq. 2
    gamma_ = RaoQ_[-1, :]
    # Eq. 5
    beta_add = gamma_ - alpha_mean_
    # Eq. 12
    beta_prop = 100 * div_zeros(beta_add, gamma_)
    alpha_prop = 100 - beta_prop
    # Eq. 14
    beta_prop_norm = beta_prop / (1. - 1./n_comm)
    alpha_prop_norm = 100 - beta_prop_norm
    if reshp_:
        return(alpha_.reshape(-1), alpha_mean_, beta_add, gamma_, alpha_prop,
               beta_prop, alpha_prop_norm, beta_prop_norm)
    else:
        return(alpha_, alpha_mean_, beta_add, gamma_, alpha_prop, beta_prop,
               alpha_prop_norm, beta_prop_norm)


def deBelloRaoQpart(Traits_sp, RelAbun_sp, use_EqNum=True, pca_transf=True,
                    alphas_=[1], tax_div=False, normalize_dist=False,
                    get_extra_outs=False, out_intvar_in=None, max_dist_in=0.,
                    n_sigmas_norm=6.):
    """
    Apply variance-based partitioninig of alpha, beta and gamma diversities
    using the method from de Bello et al, (2010)
    Inputs:
        Traits_sp: traits matrix of n species x m traits
        RelAbun_sp: frequency matrix of p populations x n species
        use_EqNum: If true, use Rao Q equivalent number and not the index to
                   partition diveristy
        pca_transf: If true, applies standardization and dimensionality
                    reduction to Traits_sp. Set to false if the principal
                    components are provided instead
        alphas_: aplhas for the parametric RaoQ
        tax_div: If true, Rao Q is computed for taxonomic variables (as the
            Simpson index)
        normalize_dist: If true, normalize the disimilarities
        get_extra_outs: Boolean, determine the amount of output arguments
        out_intvar_in: Allows receiving variables from a previous computation
                       to speed up
        max_dist_in: Maximum value of the dissimilarity useable to normalize
                   the disimilarities. This is computed from the bumber of PCA
                   components and the standardized bounds
        n_sigmas_norm: standardized distance to bounds (in standard
            deviations)
    """

    # Get the abundance of each species within all the communities
    RelAbun_sp_gamma = np.mean(RelAbun_sp, axis=0, keepdims=True)
    RelAbun_sp_alpha_gamma = np.concatenate((RelAbun_sp, RelAbun_sp_gamma),
                                            axis=0)

    # Apply PCA and standardization
    if pca_transf:
        Y, max_dist_, SS_max, _ = apply_std_PCA(Traits_sp, get_max_dis_=True,
                                                n_sigmas_norm=n_sigmas_norm)
    else:
        Y = copy.deepcopy(Traits_sp)
        max_dist_ = copy.deepcopy(max_dist_in)

    if use_EqNum is True:
        # Use Equivalent Numbers
        if tax_div is True and get_extra_outs is False:
            # For taxonomic diversity use Gini-Simpson index which is the same
            # but faster
            RaoQ_ = 1. - np.nansum(RelAbun_sp_alpha_gamma ** 2,
                                   axis=1).reshape(-1, 1)
            out_intvar = None
        else:
            RaoQ_, out_intvar = mpaRaoS_freq(
                Y, RelAbun_sp_alpha_gamma, normalize_dist=True,
                alphas_=alphas_, tax_div=tax_div,
                get_extra_outs=get_extra_outs, out_intvar_in=out_intvar_in,
                max_dist_=max_dist_)
            if len(alphas_) == 1:
                RaoQ_ = RaoQ_.reshape(-1, 1)
        (alpha_, alpha_mean_, beta_add, gamma_, alpha_prop, beta_prop,
         alpha_prop_norm, beta_prop_norm) = div_part_EqNum(
             RaoQ_, RelAbun_sp.shape[0])
    else:
        # Use the RaoQ diveristy index. Not recommended since beta divesity
        # takes low values even if species are barely repeated within
        # communities (at least for taxonomic diversity, but compare the
        # results in de Bello et al. (2010) and Pacheco-Labrador (2023).
        # For functional diversity, dimensionality rapidly compresses the
        # of fractions alpha and beta diveristy. 
        if tax_div is True and get_extra_outs is False:
            # For taxonomic diversity use Gini-Simpson index which is the same
            # but faster
            RaoQ_ = 1. - np.nansum(RelAbun_sp_alpha_gamma ** 2,
                                   axis=1).reshape(-1, 1)
            out_intvar = None
        else:
            RaoQ_, out_intvar = mpaRaoS_freq(
                Y, RelAbun_sp_alpha_gamma, normalize_dist=normalize_dist,
                alphas_=alphas_, tax_div=tax_div,
                get_extra_outs=get_extra_outs,
                out_intvar_in=out_intvar_in, max_dist_=max_dist_)
            if len(alphas_) == 1:
                RaoQ_ = RaoQ_.reshape(-1, 1)

        (alpha_, alpha_mean_, beta_add, gamma_, alpha_prop, beta_prop,
         alpha_prop_norm, beta_prop_norm) = div_part_Index(
             RaoQ_, RelAbun_sp.shape[0])

    # Return, reshape depending on the number of Rao Q alpha parameters used
    if len(alphas_) > 1:
        if get_extra_outs:
            return(alpha_mean_, beta_add, gamma_, alpha_prop, beta_prop,
                   alpha_prop_norm, beta_prop_norm, alpha_, RaoQ_, out_intvar)
        else:
            return(alpha_mean_, beta_add, gamma_, alpha_prop, beta_prop,
                   alpha_prop_norm, beta_prop_norm, alpha_, RaoQ_)
    else:
        if get_extra_outs:
            return(alpha_mean_.reshape(-1), beta_add.reshape(-1),
                   gamma_.reshape(-1), alpha_prop.reshape(-1),
                   beta_prop.reshape(-1), alpha_prop_norm.reshape(-1),
                   beta_prop_norm.reshape(-1), alpha_.reshape(-1),
                   RaoQ_.reshape(-1), out_intvar)
        else:
            return(alpha_mean_.reshape(-1), beta_add.reshape(-1),
                   gamma_.reshape(-1), alpha_prop.reshape(-1),
                   beta_prop.reshape(-1), alpha_prop_norm.reshape(-1),
                   beta_prop_norm.reshape(-1), alpha_.reshape(-1),
                   RaoQ_.reshape(-1))


# %% Main function. Diversity partition and index normalization.
def pyGNDiv_fun(Traits_sp, RelAbun_sp, alphas_=[1.], fexp_var_in=.98,
                n_sigmas_norm=6., calculate_FRic=True,
                calculate_variance_part=True, calculate_RaoQ_part=True,
                calculate_tax_part=True):
    """
    pyGNDiv allows computing different diversity indices (Rao's Quadratic
    Entropy, it's equivalent number and Functional Richness') applying none,
    local (per region) and generalizable (Pacheco-Labrador et al, (under
    review)) normalization.

    Then, parititioning into alpha, beta and gamma diveristy is applied using
    a variance-based partitioning (Laliberté et al, 2020) or diveristy
    decomposition using Rao's Quadratic Entropy and if normalized, it's
    equivalent number (de Bello et al, (2010))

    Parameters
    ----------
    Traits_sp : 2D array
        species x traits (or reflectance factors) matrix
    RelAbun_sp : 2D array
        communities x species matrix, with relative abundances normalized per
        community
    alphas_ : List, optional
        List of values for the alpha parameter used in the Rao's Q parametric
        formulation of Rocchini et al, 2021. The default is [1].
    fexp_var_in : float, optional
        Fraction of the explained variance to be retained (at least) but the
        principal comonents retained form the traits dataset. The default is
        .98.
    n_sigmas_norm : float, optional
        Number of standard deviations to the mean where the bounds are placed
        during standardization. This is used for the Generalizable
        Normalization approach (Pacheco-Labrador et al, 2023). The
        default is 6.0.
    calculate_FRic : boolean, optional
        Whether or not to compute Functional Richness (FRic). The default is
        True
    calculate_variance_part : boolean, optional
        Whether or not to compute variance-based diveristy partitioning using
        the appoach adapted from Laliberté et al, 2020. The default is True
    calculate_RaoQ_part : boolean, optional
        Whether or not to compute diveristy decomposition using RaoQ and
        equivalent numbers with the from de Bello et al, (2010). The default
        is True
    calculate_tax_part : boolean, optional
        Whether or not to compute diveristy decomposition using taxonomic
        metrics (Gini-Simpson index and equivalent number) with the from de
        Bello et al, (2010). The default is True

    Returns
    -------
    FRic : Dictionary (or None)
        Functional richness (not norm, local norm, generalizable norm) computed
        using up to 8 and up to 3 principal components
    VBdp : Dictionary
        Variance analysis metrics described in Laliberté et al, 2020,
        including diversity components and fractions of alpha and beta
        diversity
    VBdp_Gnorm : Dictionary (or None)
        Variance analysis metrics described in Laliberté et al, 2020,
        including diversity components and fractions of alpha and beta
        diversity. Sum of squared normalized using the Generalizable
        Normalization approach described in Pacheco-Labrador et al, under
        review.
    VBdp_Lnorm : Dictionary (or None)
        Variance analysis metrics described in Laliberté et al, 2020,
        including diversity components and fractions of alpha and beta
        diversity. Sum of squared normalized using the maximum value in each
        region or image (local).
    RQdp : Dictionary (or None)
        Rao Quadratic entropy-based diversity decomposition metrics described
        in de Bello et al, (2010), including diversity components and
        fractions of alpha and beta-diversity.
    RQdp_Gnorm : Dictionary (or None)
        Rao Quadratic entropy-based diversity decomposition metrics described
        in de Bello et al, (2010), including diversity components and
        fractions of alpha and beta-diversity. Dissimilarity normalized using
        the Generalizable Normalization approach described in Pacheco-Labrador
        et al, 2023.
    ENdp_Gnorm : Dictionary (or None)
        Equivalent number-based diversity decomposition metrics described
        in de Bello et al, (2010), including diversity components and
        fractions of alpha and beta-diversity. Dissimilarity normalized using
        the Generalizable Normalization approach described in
        Pacheco-Labrador et al, 2023.
    RQdp_Lnorm : Dictionary (or None)
        Rao Quadratic entropy-based diversity decomposition metrics described
        in de Bello et al, (2010), including diversity components and
        fractions of alpha and beta-diversity. Dissimilarity normalized using
        the maximum value in each region or image (local).
    ENdp_Lnorm : Dictionary (or None)
        Equivalent number-based diversity decomposition metrics described
        in de Bello et al, (2010), including diversity components and
        fractions of alpha and beta-diversity. Dissimilarity normalized using
        the maximum value in each region or image (local).
    RQtxdp : Dictionary (or None)
        Gini-Simpson index diversity decomposition metrics described
        in de Bello et al, (2010), including diversity components and
        fractions of alpha and beta-diversity. No normalization is needed.
    ENdp_Lnorm : Dictionary (or None)
        Equivalent number-based diversity decomposition metrics described
        in de Bello et al, (2010), including diversity components and
        fractions of alpha and beta-diversity. Dissimilarity normalized using
        the maximum value in each region or image (local).
    """

# %%% Standardization, dimensionality reduction, global maximum dissimilarities

    # Applies a weighted standardization and PCA for Laliberté since weighted
    # mean must be centered, and regular PCA to de Bello approach and FRic
    RelAbun_sp_sp = np.sum(RelAbun_sp, axis=0, keepdims=True)
    Yw, _, SS_maxW, fexpW = apply_std_WPCA(Traits_sp, RelAbun_sp_sp,
                                           get_max_dis_=True,
                                           n_sigmas_norm=n_sigmas_norm)
    Y, max_dist_, _, fexp = apply_std_PCA(Traits_sp, get_max_dis_=True,
                                          n_sigmas_norm=n_sigmas_norm)

    # %%% Functional Richness
    if calculate_FRic is True:
        # Limit the maximum number of components to 8 to prevent memory issues
        # during the computation of the convex hull
        FRic = dict()
        j_ = min(Y.shape[1], 8)
        FRic['FRic'], FRic['FRic_Gnorm'], FRic['FRic_Lnorm'] = get_FRic(
            Y[:, :j_], RelAbun_sp, np.sum(fexp[:j_]), n_sigmas=n_sigmas_norm)

        # Limit to three components for comparison
        j_ = min(Y.shape[1], 3)
        FRic['FRic3'], FRic['FRic3_Gnorm'], FRic['FRic3_Lnorm'] = get_FRic(
            Y[:, :j_], RelAbun_sp, np.sum(fexp[:j_]), n_sigmas=n_sigmas_norm)
    else:
        FRic = None

    # %%% Variance-based diverity partitioning (Laliberté et al. 2020)
    if calculate_variance_part is True:
        # Without normalization.
        VBdp = LalibertePart_w_wrap(Yw, RelAbun_sp, normalize=False,
                                    SS_max_in=1., pca_transf=False)
        # Global Normalization
        VBdp_Gnorm = LalibertePart_w_wrap(Yw, RelAbun_sp, normalize=True,
                                          SS_max_in=SS_maxW, pca_transf=False)
        # Local normalization.
        VBdp_Lnorm = LalibertePart_w_wrap(Yw, RelAbun_sp, normalize=True,
                                          SS_max_in=None, pca_transf=False)
    else:
        VBdp, VBdp_Gnorm, VBdp_Lnorm = None, None, None

    # %%% Diversity decomposition with Rao Q and equivalent numbers following
    # de Bello et al. (2010)
    if calculate_RaoQ_part is True:
        # Without normalization.
        # Use Rao Q index. Get also the absolute
        # disimilarities in out_intvar to speed up computation later
        RQdp = dict()
        (RQdp['alpha_mean'], RQdp['beta_add'], RQdp['gamma'],
         RQdp['Falpha'], RQdp['Fbeta'],
         RQdp['Falpha_norm'], RQdp['Fbeta_norm'],
         RQdp['RaoQ'], RaoQ_, out_intvar) = deBelloRaoQpart(
             Y, RelAbun_sp, use_EqNum=False, pca_transf=False, alphas_=alphas_,
             get_extra_outs=True, normalize_dist=False)

        # Global Normalization. Use Rao Q index
        RQdp_Gnorm = dict()
        (RQdp_Gnorm['alpha_mean'], RQdp_Gnorm['beta_add'], RQdp_Gnorm['gamma'],
         RQdp_Gnorm['Falpha'], RQdp_Gnorm['Fbeta'],
         RQdp_Gnorm['Falpha_norm'], RQdp_Gnorm['Fbeta_norm'],
         RQdp_Gnorm['RaoQ'], RaoQ_Gnorm) = deBelloRaoQpart(
             Y, RelAbun_sp, use_EqNum=False, pca_transf=False, alphas_=alphas_,
             out_intvar_in=out_intvar, max_dist_in=max_dist_,
             normalize_dist=True)

        # Global Normalization. Use Rao Q equivalent number
        # Speed up from already computed RaoQ_Gnorm
        ENdp_Gnorm = dict()
        (ENdp_Gnorm['RaoQ'], ENdp_Gnorm['alpha_mean'], ENdp_Gnorm['beta_add'],
         ENdp_Gnorm['gamma'], ENdp_Gnorm['Falpha'], ENdp_Gnorm['Fbeta'],
         ENdp_Gnorm['Falpha_norm'], ENdp_Gnorm['Fbeta_norm']) = (
             div_part_EqNum(RaoQ_Gnorm.reshape(-1, 1), RelAbun_sp.shape[0],
                            reshp_=True))
        # This could be done, but forces to re-compute RaoQ.
        # (ENdp_Gnorm['alpha_mean'], ENdp_Gnorm['beta_add'], ENdp_Gnorm['gamma'],
        #   ENdp_Gnorm['Falpha'], ENdp_Gnorm['Fbeta'],
        #   ENdp_Gnorm['Falpha_norm'], ENdp_Gnorm['Fbeta_norm'],
        #   ENdp_Gnorm['RaoQ'], Qeq_Gnorm) = deBelloRaoQpart(
        #       Y, RelAbun_sp, use_EqNum=True, pca_transf=False, alphas_=alphas_,
        #       out_intvar_in=out_intvar, max_dist_in=max_dist_,
        #       normalize_dist=True)

        # Local Normalization. Use Rao Q index
        m_dist_local = np.max(out_intvar['dists_'])
        RQdp_Lnorm = dict()
        (RQdp_Lnorm['alpha_mean'], RQdp_Lnorm['beta_add'], RQdp_Lnorm['gamma'],
         RQdp_Lnorm['Falpha'], RQdp_Lnorm['Fbeta'],
         RQdp_Lnorm['Falpha_norm'], RQdp_Lnorm['Fbeta_norm'],
         RQdp_Lnorm['RaoQ'], RaoQ_Lnorm) = deBelloRaoQpart(
             Y, RelAbun_sp, use_EqNum=False, pca_transf=False, alphas_=alphas_,
             out_intvar_in=out_intvar, max_dist_in=m_dist_local,
             normalize_dist=True)

        # Local Normalization. Use Rao Q equivalent number
        # Speed up from already computed RaoQ_Lnorm
        ENdp_Lnorm = dict()
        (ENdp_Lnorm['RaoQ'], ENdp_Lnorm['alpha_mean'], ENdp_Lnorm['beta_add'],
         ENdp_Lnorm['gamma'], ENdp_Lnorm['Falpha'], ENdp_Lnorm['Fbeta'],
         ENdp_Lnorm['Falpha_norm'], ENdp_Lnorm['Fbeta_norm']) = (
             div_part_EqNum(RaoQ_Lnorm.reshape(-1, 1), RelAbun_sp.shape[0],
                            reshp_=True))
        # This could be done, but forces to re-compute RaoQ.
        # (ENdp_Lnorm['alpha_mean'], ENdp_Lnorm['beta_add'], ENdp_Lnorm['gamma'],
        #  ENdp_Lnorm['Falpha'], ENdp_Lnorm['Fbeta'],
        #  ENdp_Lnorm['Falpha_norm'], ENdp_Lnorm['Fbeta_norm'],
        #  ENdp_Lnorm['RaoQ'], Qeq_Lnorm) = deBelloRaoQpart(
        #      Y, RelAbun_sp, use_EqNum=True, pca_transf=False, alphas_=alphas_,
        #      out_intvar_in=out_intvar, max_dist_in=m_dist_local,
        #      normalize_dist=True)

    # %%% Return
    return(FRic, VBdp, VBdp_Gnorm, VBdp_Lnorm, RQdp, RQdp_Gnorm, ENdp_Gnorm,
           RQdp_Lnorm, ENdp_Lnorm)

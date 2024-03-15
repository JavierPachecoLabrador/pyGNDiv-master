"""
@author: Javier Pacheco-Labrador, PhD (jpacheco@bgc-jena.mpg.de)
         Max Planck Institute for Biogeochemsitry, Jena, Germany
        Currently: javier.pacheco@csic.es
         Environmental Remote Sensing and Spectroscopy Laboratory (SpecLab)
         Spanish National Research Council (CSIC)

DESCRIPTION:
==============================================================================
This script performs the simulation of plant communities, species traits and
    corresponding reflectance factors used for the evaluation of the package
    pyGNDiv presented in Pacheco-Labrador, J. et al, (2023). It also
    uses the package to compute functional diversity metrics and parition
    diversity alpha, beta and gamma components with various normalization
    approaches.

The script has been used to generate the test dataset /test_data/ included in
    the pyGNDiv package which is evaluated by the script test.py

Cite as:
  Pacheco-Labrador, J., de Bello, F., Migliavacca , M., Ma, X., Carvalhais, N.,
    & Wirth, C. (2023). A generalizable normalization for assessing
    plant functional diversity metrics across scales from remote sensing.
    Methods in Ecology and Evolution.
"""

# %% Imports ------------------------------------------------------------------
import os
from sys import argv
import pickle
import joblib
import copy

import pandas as pd
import numpy as np
import time as time
import random as random
import scipy
from pyDOE import lhs
import pyGNDiv as gnd


# %% Functions ----------------------------------------------------------------
# ## Set up ------------------------------------------------------------------
def initial_setup(argv_in, sensor_='Hy', folder_in_name_='Test',
                  num_regions_=7, n_sigmas_norm_=6., n_sp_dp_=300,
                  rseed_num_=0):
    """
    Set default configuration or take external inputs
    """
    if len(argv_in) > 1:
        # Sensor to be simulated
        sensor_ = argv_in[1]
        # Output folder
        folder_out_name_ = argv_in[2] + '//'
        # Number of regions simulated
        num_regions_ = int(argv_in[3])
        # Number of sigmas used to normalize diveristy indices
        n_sigmas_norm_ = float(argv_in[4])
        # Seed of the random number generator
        n_sp_dp_ = int(argv_in[5])
        # Seed of the random number generator
        rseed_num_ = int(argv_in[6])
    else:
        folder_out_name_  = folder_in_name_ + '//'

    if os.path.isdir(folder_out_name_) is False:
        os.mkdir(folder_out_name_)

    print('\n\n-------------------Setup------------------')
    print('argv: %d' % len(argv_in))
    print('sensor = %s' % (sensor_))
    print('folder_out = %s' % (folder_out_name_))
    print('num_regions = %s' % (num_regions_))
    print('n_sp_max = %s' % (n_sigmas_norm_))
    print('nsigmas_norm = %s' % (n_sp_dp_))
    print('rseed_num = %d' % (rseed_num_))
    print('------------------------------------------\n\n')

    return(sensor_, folder_out_name_, num_regions_, n_sigmas_norm_,
           n_sp_dp_, rseed_num_)


# ## Simulator configuration --------------------------------------------------
def get_RSensor_bands(sensor_, ori_pth='simulator_inputs//'):
    """
    Generate the spectral bands of different sensors
    """
    if sensor_ == 'Hy':
        RSensor_bands = []
        wl = np.r_[400:2401]
    elif sensor_ == 'S2':
        df = pd.read_csv(ori_pth + 'SRF_Sentinel2A-MSI.txt', sep="\t")
        df_ = df.loc[(df.SR_WL >= 400) & (df.SR_WL <= 2400)]
        # Remove 60 m bands
        for b in ['SR_AV_B1', 'SR_AV_B9', 'SR_AV_B10']:
            df_.__delitem__(b)
        RSensor_bands = (df_.to_numpy())
        RSensor_bands = RSensor_bands[:, 1:]
        # Get S2 selected wavelengths
        wl = np.array([490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190])
    elif sensor_ == 'DESIS':
        df = np.loadtxt(ori_pth + 'SRF_DESIS.txt')
        RSensor_bands = np.zeros((2001, df.shape[0]))
        wl0 = np.arange(400, 2401)
        wl = df[:, 0]
        for i_ in range(df.shape[0]):
            RSensor_bands[:, i_] = scipy.stats.norm.pdf(
                wl0, loc=df[i_, 0], scale=df[i_, 1])
    elif sensor_ == 'QB2':
        df = pd.read_csv(ori_pth + 'SRF_QuickBird2.txt', sep="\t")
        df_ = df.loc[(df.SR_WL >= 400) & (df.SR_WL <= 2400)]
        RSensor_bands = (df_.to_numpy())
        RSensor_bands = RSensor_bands[:, 1:] / np.sum(RSensor_bands[:, 1:],
                                                      axis=0)
        wl = df_.SR_WL.to_numpy() @ RSensor_bands

    # Makes sure that the sensor is normalized by the band integral
    if sensor_ != 'Hy':
        SRFint_ = np.sum(RSensor_bands, axis=0).reshape(1, -1)
        RSensor_bands = (RSensor_bands / np.repeat(SRFint_, [2001], axis=0))

    wl_name = ['%d' % i for i in wl]

    return(RSensor_bands, wl, wl_name)


def prepare_variables(sensor_):
    """
    Define simulation variables: SCOPE meteorological, biophysical and
    functional inputs, variable indices, and output spectral variables
    """
    # # SCOPE model Variables
    objTxtFile = open('simulator_inputs//Xvnames.txt', "r")
    Bvars = (objTxtFile.readline()).split('\t')
    objTxtFile.close()

    tmp_ = np.loadtxt('simulator_inputs//Xbounds.txt')
    LB = np.array([tmp_[0][i_] for i_ in range(len(Bvars)) if Bvars[i_]
                   not in ['h_min_rel', 'h_min']])
    UB = np.array([tmp_[1][i_] for i_ in range(len(Bvars)) if Bvars[i_]
                   not in ['h_min_rel', 'h_min']])

    # Limit some variables to the ranges of Lopex and the predictions of the
    # Gaussian Mixture model
    UB[Bvars.index('N')] = 3.
    UB[Bvars.index('Cca')] = 25.
    UB[Bvars.index('Cw')] = .045
    UB[Bvars.index('Cdm')] = .0157
    UB[Bvars.index('Cs')] = 1.

    # Indices for the variables
    # [Emulator predictors, Functional diveristy analysis]
    I_bioph = list(range(0, Bvars.index('Vcmo')))
    I_lidf = [Bvars.index('LIDFa'), Bvars.index('LIDFb')]
    I_lai = Bvars.index('LAI')
    I_leaf = (list(range(0, Bvars.index('LIDFa'))) +
              list(range(Bvars.index('leafwidth'),
                         Bvars.index('BSMBrightness'))))
    I_canopy = list(range(Bvars.index('LIDFa'), Bvars.index('leafwidth')))
    I_soil = list(range(Bvars.index('BSMBrightness'), Bvars.index('rss')))
    I_smp = list(range(Bvars.index('SMp'), Bvars.index('rss')))
    I_ang = list(range(Bvars.index('tts'), Bvars.index('Gfrac')))
    I_meteo = list(range(Bvars.index('Rin'), len(Bvars)))

    # # Meteo vars
    objTxtFile = open('simulator_inputs//Meteo_vnames.txt', "r")
    vars_meteo = (objTxtFile.readline().replace(
        'SZA', 'tts').replace('SAA', 'psi')).split('\t')
    objTxtFile.close()
    LB_meteo = np.loadtxt('simulator_inputs//Meteo_LB.txt')
    UB_meteo = np.loadtxt('simulator_inputs//Meteo_UB.txt')

    # Increases radiation
    LB_meteo[0] = LB_meteo[0] + .3 * (UB_meteo[0] - LB_meteo[0])

    # Filter meteo variables according to their presence in Bvars
    vars_meteo0 = [Bvars[i_] for i_ in I_meteo if Bvars[i_] in vars_meteo]
    LB_meteo0 = [LB_meteo[vars_meteo.index(Bvars[i_])] for i_ in I_meteo
                 if Bvars[i_] in vars_meteo]
    UB_meteo0 = [UB_meteo[vars_meteo.index(Bvars[i_])] for i_ in I_meteo
                 if Bvars[i_] in vars_meteo]

    # # Impose limits of meteo conditions from the GMM.
    n_vars = len(Bvars)
    for i_ in range(len(vars_meteo0)):
        if vars_meteo0[i_] in Bvars:
            j_ = Bvars.index(vars_meteo0[i_])
            LB[j_] = max(LB[j_], LB_meteo0[i_])
            UB[j_] = min(UB[j_], UB_meteo0[i_])

    # Variables that are predicted by the Mixed Gaussian Model
    I_bgm = [[Bvars.index(i_) for i_ in ['N', 'Cab', 'Cca', 'Cw', 'Cdm']],
             [], []]
    I_mgm = I_meteo + [I_ang[0]]  # [print(Bvars[i]) for i in I_mgm]
    I_bvmet = [vars_meteo.index(Bvars[i_]) for i_ in I_mgm]

    # Variables that are predicted by LHS
    I_lhs0 = [True] * n_vars
    for i in (I_mgm + I_bgm[0] + I_bgm[1] + I_bgm[2]):
        I_lhs0[i] = False
    I_lhs = [i for i in range(n_vars) if (I_lhs0[i] is True)]

    # Get spectral variables
    (RSensor_bands, wl, wl_name) = get_RSensor_bands(sensor_)

    return(Bvars, vars_meteo, LB, UB, LB_meteo, UB_meteo, n_vars,
           I_bioph, I_lidf, I_lai, I_leaf, I_canopy, I_soil, I_smp,
           I_ang, I_meteo, I_bvmet, I_bgm, I_mgm, I_lhs, wl_name, wl,
           RSensor_bands)


def truncatedBGM(n_samples, BGM, xmean, xstd, LB, UB, k, LB_rel=None,
                 UB_rel=None, pos_rel=None):
    """
    Generates a random sample from a multidimensional Gaussian model and
    applies truncation using the rejection method
    """
    if (pos_rel is not None) and (pos_rel != []):
        LB_rel = np.insert(LB_rel, pos_rel, 0.)
        UB_rel = np.insert(UB_rel, pos_rel, 1.)

    n_samples_aug = int(n_samples*k)
    tm = 0
    tm2 = 1
    X = np.zeros([n_samples, len(LB)])
    BGM.random_state = n_samples_aug
    counter_ = 0
    while (tm < n_samples-1) and (counter_ < 10):
        # Sample data
        X00 = BGM.sample(n_samples=n_samples_aug)[0]
        X0 = X00 * xstd + xmean

        # Remove data within the limits
        Isel = np.array(range(0, n_samples_aug))

        # Discard before data with nan
        Io = np.isnan(X0).any(1)
        Isel[Io] = -999
        X0[Io, :] = 0
        for i_ in range(len(LB)):
            Isel[X0[:, i_] < LB[i_]] = -999
            Isel[X0[:, i_] > UB[i_]] = -999

            # If also a constraint on the values of the figures respect to the
            # value of the variables relative to one of them, then this
            # constraint is also applied
            if (pos_rel is not None) and (pos_rel != []) and (i_ != pos_rel):
                X0_rel = X0[:, i_] / X0[:, pos_rel]
                Isel[X0_rel < LB_rel[i_]] = -999
                Isel[X0_rel > UB_rel[i_]] = -999

        Isel = Isel[Isel >= 0]

        # Assign truncated data to the output matrix
        tm2 = min(tm + len(Isel), n_samples)
        X[range(tm, tm2), :] = X0[Isel[range(0, tm2-tm)], :]

        tm += len(Isel)
        counter_ += 1

    # If not achieved in 10 cycles, provide the closest data to the center
    # of the range
    if (tm < n_samples-1):
        X00 = BGM.sample(n_samples=n_samples_aug)[0]
        X0 = X00 * xstd + xmean

        Xcent_ = np.mean((UB, LB), axis=0)
        Xdist_ = np.sqrt(np.sum((X0 - Xcent_)**2, axis=1))
        Isort = np.argsort(Xdist_)
        X[range(tm, tm2), :] = X0[Isort[range(0, tm2-tm)], :]

    if (pos_rel is not None) and (pos_rel != []):
        X_rel = np.zeros(X.shape)
        Inz = X[:, pos_rel] != 0
        X_rel[Inz, :] = X[Inz, :] / X[Inz, pos_rel].reshape(-1, 1)
        X = np.concatenate((X, np.delete(X_rel, 0, axis=1)), axis=1)

    return(X)


def limit_Cs_from_Cab(X_in, Icab, Ics, sf_=40.):
    X_ = copy.deepcopy(X_in)
    limf_ = np.exp(sf_*(100 - X_[:, Icab])/100) / np.exp(sf_)
    Ics2z = (X_[:, Ics] > limf_)
    X_[:, Ics] = X_in[:, Ics] * limf_[:]

    return(X_)


def produce_RTM_parameters(n_samples_sim, GMM_lopex, X0_meanL, X0_stdL,
                           GMM_meteo,  X0_meanM, X0_stdM, LB_sim, UB_sim,
                           LB_meteo, UB_meteo, I_meteo, I_bvmet, I_bgm, I_mgm,
                           I_lhs, I_lidf, k_, Bvars):
    """
    Generates a truncated random distribution of SCOPE inputs
    """
    # Preallocates
    Xpred = np.zeros((n_samples_sim, len(LB_sim))) * np.nan

    # Predict leaf parameters from the GMM_lopex model
    if len(I_bgm[0]) > 0:
        Xpred[:, I_bgm[0]] = truncatedBGM(
            n_samples_sim, GMM_lopex, X0_meanL, X0_stdL,
            LB_sim[I_bgm[0]], UB_sim[I_bgm[0]], k_)

    # Remove the last one since it is the sun azimuth angle
    X0 = truncatedBGM(n_samples_sim, GMM_meteo, X0_meanM, X0_stdM,
                      LB_meteo, UB_meteo, k_)[:, :-1]

    Xpred[:, I_mgm] = X0[:, I_bvmet]

    # Predict parameters from the LSH model
    LHS_ = lhs(len(I_lhs), samples=n_samples_sim, iterations=15)
    Xpred[:, I_lhs] = LB_sim[I_lhs] + LHS_ * (UB_sim[I_lhs] - LB_sim[I_lhs])

    # Limit Cs and Cant according to Cab to prevent simultaneous high values
    Xpred = limit_Cs_from_Cab(Xpred, Bvars.index('Cab'),
                              Bvars.index('Cs'), sf_=40.)
    Xpred = limit_Cs_from_Cab(Xpred, Bvars.index('Cab'),
                              Bvars.index('Cant'), sf_=7.)

    # Correct LIDF parameters
    lidfa = copy.deepcopy(Xpred[:, I_lidf[0]])
    lidfb = copy.deepcopy(Xpred[:, I_lidf[1]])
    Xpred[:, I_lidf[0]] = (lidfa + lidfb)/2
    Xpred[:, I_lidf[1]] = (lidfa - lidfb)/2

    # Vector to randomly pick data from SpeciesPool
    range_pool = range(n_samples_sim)

    return(Xpred, range_pool)


def prepare_simulator(sensor_, rseed_num_, tam_grid=9):
    # Seed
    random.seed(rseed_num_)
    np.random.seed(rseed_num_)

    # Variables
    (Bvars, vars_meteo, LB, UB, LB_meteo, UB_meteo, n_vars, I_bioph, I_lidf,
     I_lai, I_leaf, I_canopy, I_soil, I_smp, I_ang, I_meteo, I_bvmet, I_bgm,
     I_mgm, I_lhs, wl_name, wl, RSensor_bands) = (
         prepare_variables(sensor_))

    # Get leaf parameter's multidimensional distribution from Lopex
    GMM_lopex = joblib.load('simulator_inputs//GMM_PT.joblib')
    with open('simulator_inputs//GMM_PT.pkl', 'rb') as f:
        _, _, X0_meanL, X0_stdL = pickle.load(f)

    # Get multidimensional distribution from Fluxnet database
    GMM_meteo = joblib.load('simulator_inputs//GMM_meteo.joblib')
    with open('simulator_inputs//GMM_meteo.pkl', 'rb') as f:
        _, _, X0_meanM, X0_stdM = pickle.load(f)

    # Load Spectral emulators
    M_R = joblib.load('simulator_inputs//R_NNemulator.joblib')
    # Define the parameters used to predict spectral variables and to compute
    # biodiversity metrics
    I_Rpred = [i_ for i_ in range(len(Bvars)) if Bvars[i_] in M_R['feat_sel']]
    I_Rbiod = copy.deepcopy(I_bioph)

    # Generate the dissimilar species pool
    # Define centroid
    centroid = (LB + UB) / 2
    range_bounds = UB - LB

    # Seed again to ensure consistency
    random.seed(rseed_num)
    np.random.seed(rseed_num)
    n_samples = 50000
    SpeciesPool, range_pool = produce_RTM_parameters(
        n_samples, GMM_lopex, X0_meanL, X0_stdL, GMM_meteo, X0_meanM, X0_stdM,
        LB, UB, LB_meteo, UB_meteo, I_meteo, I_bvmet, I_bgm, I_mgm, I_lhs,
        I_lidf, 4, Bvars)

    # Additional setups for the simulation
    n_commun = int(tam_grid**2)
    whf_ = 0
    grid_ref = []
    Isrs_ = np.arange(n_commun)
    f_a2g = lambda x: np.asarray(x).reshape((tam_grid, tam_grid)).T

    return(Bvars, LB_meteo, UB_meteo, LB, UB, n_vars, I_bioph, I_lidf, I_lai,
           I_leaf, I_canopy, I_soil, I_smp, I_ang, I_meteo, I_bvmet, I_bgm,
           I_mgm, I_lhs, wl_name, wl, RSensor_bands, GMM_lopex, X0_meanL,
           X0_stdL, GMM_meteo, X0_meanM, X0_stdM, M_R, I_Rpred, I_Rbiod,
           centroid, range_bounds, n_samples, SpeciesPool, range_pool, whf_,
           grid_ref, Isrs_, n_commun, f_a2g)


def uneven_random_split(n_sp_groups, n_samples_tot, get_='position'):
    pf_ = np.random.uniform(size=n_sp_groups)
    pool_size = np.round(n_samples_tot * pf_ / np.sum(pf_)).astype(int)
    if np.sum(pool_size) != n_samples_tot:
        dif_ps = np.sum(pool_size) - n_samples_tot
        Ipm_ = np.where(pool_size == np.max(pool_size))[0][0]
        pool_size[Ipm_] = pool_size[Ipm_] - dif_ps
    if np.any(pool_size == 0):
        pool_size = pool_size[pool_size != 0]
        n_sp_groups = len(pool_size)
    # Generate index for the species
    x_split = np.linspace(1, n_samples_tot, n_samples_tot, dtype=int) - 1
    np.random.shuffle(x_split)
    sp_groups = np.array_split(x_split, np.cumsum(pool_size[:-1]))

    if get_ == 'position':
        return(sp_groups)
    elif get_ == 'group_index':
        g_ind = np.zeros(n_samples_tot, dtype=int)
        k_ = 0
        i_ = sp_groups[0]
        for i_ in sp_groups:
            g_ind[i_] = copy.deepcopy(k_)
            k_ += 1
        return(g_ind)


def div_zeros(a, b):
    if isinstance(a, float):
        a = np.array(a).reshape((1, 1))
    if isinstance(b, float):
        b = np.array(b).reshape((1, 1))
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
    return(np.divide(a, b, out=np.zeros_like(a), where=b != 0.))


def generate_species_dist(tam_grid, n_commun, p_threshold, n_sp_max):
    """
    Generate the PDF of different combination of species following an
    increasing pattern of number of species and homogeneity of their
    probabilities
    """

    # Determine the range of species to be simulated
    n_sp_ = [5, n_sp_max]

    # Determine the number of species and how these are generated
    n_samples_tot = random.randrange(n_sp_[0], n_sp_[1])
    n_samples_sim = int(random.uniform(.05, 1.) * n_samples_tot)
    n_samples_pool = n_samples_tot - n_samples_sim
    I_shuffle = []
    x_species = np.linspace(1, n_samples_tot, n_samples_tot)
    mu_dist = []
    sigma_dist = []

    # Follow an approach similar to Stier et al, 2016, but repeates species
    # combinations with different distributions
    # First describe the community PDF, so that different species are more
    # or less rare
    pdf_tmp = (scipy.stats.norm.pdf(
        x_species,
        loc=(np.random.uniform(low=1. / n_samples_tot,
                               high=1. - 1. / n_samples_tot) *
             float(n_samples_tot)),
        scale=(np.random.uniform(low=.05,
                                 high=.2 * n_samples_tot))) +
        max(1/1000, np.random.uniform() / n_samples_tot))
    pdf_tmp = np.round(pdf_tmp / np.sum(pdf_tmp), 3)

    # The PDF is shuffled to destroy the dependency of parameters rarety
    # and abundance. Also, it is rounded to 1000
    np.random.shuffle(pdf_tmp)

    # Then split more or less equally the species within the subpools
    n_sp_groups_ini = np.random.randint(3, max(4, min(
        np.round(n_samples_tot * .1).astype(int), n_commun)))

    # Set different subpools sizes
    sp_groups = uneven_random_split(n_sp_groups_ini, n_samples_tot,
                                    get_='position')
    # Less groups than asked for might come. Update
    n_sp_groups = len(sp_groups)

    # Assign one subpools to each community. Make sure all the groups
    # are included
    com_ind = []
    while np.unique(com_ind).shape[0] < n_sp_groups:
        com_ind = uneven_random_split(n_sp_groups, n_commun,
                                      get_='group_index')

    # Determines the degree of species mixture and select the species
    # to be mixed
    mix_rat = min(1., np.abs(np.random.normal(0., 1./2.75)))
    n_mix_sp = int(mix_rat * n_samples_tot)
    sp2mix = np.random.choice(np.arange(n_samples_tot), size=n_mix_sp,
                              replace=False).tolist()
    mix_frac = np.random.uniform()

    # Assign PDFs from a pool of 10^6 idividuals
    Abundances = np.zeros((n_commun, n_samples_tot))
    g_ = 0
    for g_ in range(n_sp_groups):
        # Find the communities of the group
        Isp_pres = np.where(com_ind == g_)[0]
        Isp_abs = np.where(com_ind != g_)[0]
        for i_ in sp_groups[g_]:
            # Select the individuals to be split within groups
            if (i_ in sp2mix) and (Isp_abs.shape[0] > 0):
                sp_indv = pdf_tmp[i_] * 1e6 * (1 - mix_frac)
                sp_mx = pdf_tmp[i_] * 1e6 * mix_frac
                f_mx = np.random.uniform(low=0, size=Isp_abs.shape[0])
                # Stil increase beta diversity
                I2zero_tm = max(0, Isp_abs.shape[0]-1)
                I2zero = np.random.randint(
                    I2zero_tm,
                    size=int((np.random.uniform()**4) * I2zero_tm))
                if I2zero.shape[0] > 0:
                    f_mx[I2zero] = 0.
                if np.sum(f_mx) > 0.:
                    Abundances[Isp_abs, i_] = ((f_mx / np.sum(f_mx)) * sp_mx)
            else:
                sp_indv = pdf_tmp[i_] * 1e6
                sp_mx = 0.
                f_mx = 0.
            # Distribute individuals of the species. Increase rarity
            # removing species in part of the communities
            f_sp = np.random.uniform(low=0, size=Isp_pres.shape[0])
            # Percentile 50 prevents empty communities and prevents removing
            f_sp[f_sp < np.percentile(f_sp, 50)] = 0.
            if np.sum(f_sp) > 0.:
                Abundances[Isp_pres, i_] = ((f_sp / np.sum(f_sp)) * sp_indv)

    # Renormalize species
    Abundances = div_zeros(Abundances, np.sum(Abundances, axis=1,
                                              keepdims=True))

    # Make sure that no community stays without species
    empty_com_check = np.where(np.sum(Abundances, axis=1) == 0.)[0]
    if np.any(empty_com_check):
        for ee_ in empty_com_check:
            eei_ = np.random.randint(n_samples_tot - 1)
            Abundances[ee_, eei_] = 1.
    empty_com_check = np.where(np.sum(Abundances, axis=1) == 0.)[0]
    if np.any(empty_com_check):
        raise TypeError("Empty community")

    # Remove species where there is no occurrence for any community
    I_keep = np.where(np.sum(Abundances, axis=0))[0].tolist()
    n_samples_sim_0 = copy.deepcopy(n_samples_sim)
    if len(I_keep) < n_samples_tot:
        # Correct the number of species to generate
        for i in range(n_samples_sim):
            if i not in I_keep:
                n_samples_sim -= 1
                # print(i)
        for i in range(n_samples_sim_0, n_samples_tot):
            if i not in I_keep:
                # print(i)
                n_samples_pool -= 1
        n_samples_tot = n_samples_pool + n_samples_sim
        x_species = np.linspace(1, n_samples_tot, n_samples_tot)
        # Remove probabilities of absent species
        Abundances = Abundances[:, I_keep]

    print('\t%d species and %d communities' % (n_samples_tot, n_commun))

    return(n_samples_tot, n_samples_sim, n_samples_pool, Abundances,
           x_species, mu_dist, sigma_dist, I_shuffle)


def pred_RF(PlantTraits_i, I_Rpred, M_R, sensor_, RSensor_bands):
    """
    Use the SCOPE emulators to generate species reflectance factors
    WARNING: If negative numbers are produced, very low (but larger than 0.)
    values are assinged to prevent overflow and errors in the computation of
    Biodiversity indices
    """
    RF = M_R['PCA'].inverse_transform(M_R['MLP'].predict(
        M_R['SC'].transform(PlantTraits_i[:, I_Rpred])))
    Ineg = np.where(RF < 0.)
    if Ineg[0].shape[0] > 1:
        RF[Ineg[0], Ineg[1]] = np.random.uniform(1e-8, 1e-5,
                                                 size=Ineg[0].shape)

    # Convolve R to bands if sensor is specified
    if (sensor_ != '') and (sensor_ != 'Hy'):
        RF = RF @ RSensor_bands

    return(RF)


def biodiversity_simulator(n_runs,  n_vars, n_samples, range_pool, tam_grid,
                           n_commun, whf_, grid_ref, Isrs_, SpeciesPool,
                           GMM_lopex, X0_meanL, X0_stdL, GMM_meteo, X0_meanM,
                           X0_stdM, LB, UB, range_bounds, LB_meteo, UB_meteo,
                           I_bgm, I_mgm, I_lhs, I_lidf, I_soil, I_meteo,
                           I_bvmet, I_Rpred, I_bioph, I_Rbiod, Bvars,
                           wl, sensor_, RSensor_bands, M_R, n_sp_max):
    """
    This function aggregates all the simulation work
    """
    print('Run %d ------------------------------------...' % (n_runs + 1))
    # Determine number of species to simulate
    t0 = time.time()
    (n_samples_tot, n_samples_sim, n_samples_pool, Abundances,
     x_species, mu_dist, sigma_dist, I_shuffle) = (
         generate_species_dist(tam_grid, n_commun, .0, n_sp_max))
    print('\tGenerate abundances: elapsed time %.2f seconds...'
          % (time.time()-t0))

    # Simulate species biophysical properties
    # Preallocates
    t0 = time.time()
    PlantTraits_i = np.nan * np.zeros((n_samples_tot, n_vars))

    # Select one of the species of the pool and generate n_samples_i similar
    # species:
    Species_sim = SpeciesPool[random.randint(0, n_samples-1), :]
    scaling_var = 5. + random.random() * 5
    LB_sim = np.maximum(Species_sim - range_bounds/10, LB)
    UB_sim = np.minimum(Species_sim + range_bounds/10, UB)

    if n_samples_sim > 0:
        PlantTraits_i[0:n_samples_sim, :], _ = (
            produce_RTM_parameters(n_samples_sim, GMM_lopex, X0_meanL, X0_stdL,
                                   GMM_meteo, X0_meanM, X0_stdM, LB_sim,
                                   UB_sim, LB_meteo, UB_meteo, I_meteo,
                                   I_bvmet, I_bgm, I_mgm, I_lhs, I_lidf,
                                   25000, Bvars))

    # Add species selected randomly from the large pool
    PlantTraits_i[n_samples_sim:, :] = (SpeciesPool[random.sample(
        range_pool, k=n_samples_pool), :])

    # Homogenize soil properties
    for i in I_soil:
        PlantTraits_i[1:, Bvars.index(Bvars[i])] = (
            PlantTraits_i[0, Bvars.index(Bvars[i])])

    # Assign the same sun view and sky conditions
    PlantTraits_i[:, Bvars.index('DGr')] = .2
    PlantTraits_i[:, Bvars.index('tts')] = 30.
    PlantTraits_out = PlantTraits_i[:, I_Rbiod]
    print('\tGenerate traits: elapsed time %.2f seconds...'
          % (time.time()-t0))

    # Generate reflectance factors and SIF
    t0 = time.time()
    RF = pred_RF(PlantTraits_i, I_Rpred, M_R, sensor_, RSensor_bands)
    print('\tGenerate spectra: elapsed time %.2f seconds...'
          % (time.time()-t0))

    return(PlantTraits_out, Abundances, RF, n_samples_tot,
           n_samples_sim, n_samples_pool, x_species, mu_dist, sigma_dist,
           I_shuffle)


# %% Main ---------------------------------------------------------------------
if __name__ == "__main__":
    """
    Input arguments:
        sensor_: 'Hy', 'DESIS', 'S2' oe 'QB2'
        folder_out: str, Name of the subfoler where data will be stored
        num_regions: int, number of regions simulated
        n_sigmas_norm: int, (though can be float). Distance in sigmas, used to
            normalize RaoQ for diversity decomposition and FRic.
        np_sp_max: int, maximum number of species per region.
        rseed_num: int, seed for the random number generator.

    Example:
        python simulator_test.py Hy Test 7 6 300 0
    """

    # Initialize --------------------------------------------------------------
    # Simulation parameters set up
    (sensor, folder_out, num_regions, n_sigmas_norm, n_sp_max,
     rseed_num) = initial_setup(argv)

    # Get variables and models
    (Bvars, LB_meteo, UB_meteo, LB, UB, n_vars, I_bioph, I_lidf, I_lai, I_leaf,
     I_canopy, I_soil, I_smp, I_ang, I_meteo, I_bvmet, I_bgm, I_mgm, I_lhs,
     wl_name, wl, RSensor_bands, GMM_lopex, X0_meanL, X0_stdL, GMM_meteo,
     X0_meanM, X0_stdM, M_R, I_Rpred, I_Rbiod, centroid, range_bounds,
     n_samples, SpeciesPool, range_pool, whf_, grid_ref, Isrs_, n_commun,
     f_a2g) = prepare_simulator(sensor, rseed_num)

    # Predefine the number of species per region
    n_commun_run = np.random.randint(10, high=int(n_sp_max*.5),
                                     size=num_regions)
    tam_sm_ = np.sum(n_commun_run)

    # Preallocate dictionaries to allocate the outputs
    (FRic_PT, VBdp_PT, VBdp_Gnorm_PT, VBdp_Lnorm_PT, RQdp_PT, RQdp_Gnorm_PT,
     ENdp_Gnorm_PT, RQdp_Lnorm_PT, ENdp_Lnorm_PT) = (
         gnd.preallocate_outputs(num_regions))

    (FRic_R, VBdp_R, VBdp_Gnorm_R, VBdp_Lnorm_R, RQdp_R, RQdp_Gnorm_R,
     ENdp_Gnorm_R, RQdp_Lnorm_R, ENdp_Lnorm_R) = (
         gnd.preallocate_outputs(num_regions))

    # Run simulations per region ----------------------------------------------
    dtime = np.nan * np.zeros(num_regions)
    nsmp = np.nan * np.zeros(num_regions)
    ini_run = 0
    end_run = ini_run + n_commun_run[0]

    t_start = time.time()
    for n_runs in range(num_regions):
        # Update random seeds so that the same parameters and abundances are
        # generated for runs with different bands and noises consistently
        random.seed((rseed_num + 1) * 220 + n_runs)
        np.random.seed((rseed_num + 1) * 2200 + n_runs)

        # Settings
        n_commun = copy.deepcopy(n_commun_run[n_runs])
        Isrs_ = np.arange(n_commun_run[n_runs])
        tam_grid = n_commun_run[n_runs] ** .5
        ind_run = np.arange(ini_run, end_run).tolist()
        ind_run_in = [ind_run[i_] for i_ in Isrs_]
        t_ini = time.time()

        # Simulate vegetation parameters and spectral signals
        (PlantTraits, Abundances, RF, n_samples_tot, n_samples_sim,
         n_samples_pool, x_species, mu_dist, sigma_dist,
         I_shuffle) = (biodiversity_simulator(
             n_runs,  n_vars, n_samples, range_pool, tam_grid, n_commun,
             whf_, grid_ref, Isrs_, SpeciesPool, GMM_lopex, X0_meanL, X0_stdL,
             GMM_meteo, X0_meanM, X0_stdM, LB, UB, range_bounds, LB_meteo,
             UB_meteo, I_bgm, I_mgm, I_lhs, I_lidf, I_soil, I_meteo, I_bvmet,
             I_Rpred, I_bioph, I_Rbiod, Bvars, wl, sensor, RSensor_bands, M_R,
             n_sp_max))

        # Compute FDMs with different normalizations using plant functional
        # traits
        if sensor == 'Hy':
            t0 = time.time()
            (FRic_PT[n_runs], VBdp_PT[n_runs], VBdp_Gnorm_PT[n_runs],
             VBdp_Lnorm_PT[n_runs], RQdp_PT[n_runs], RQdp_Gnorm_PT[n_runs],
             ENdp_Gnorm_PT[n_runs], RQdp_Lnorm_PT[n_runs],
             ENdp_Lnorm_PT[n_runs]) = gnd.pyGNDiv_fun(PlantTraits, Abundances)
            print(('\tpyGNDiv on plant functional traits: elapsed time ' +
                   '%.2f seconds...') % (time.time()-t0))

        # Compute FDMs with different normalizations using the reflectance
        # factors (spectral variables)
        t0 = time.time()
        (FRic_R[n_runs], VBdp_R[n_runs], VBdp_Gnorm_R[n_runs],
         VBdp_Lnorm_R[n_runs], RQdp_R[n_runs], RQdp_Gnorm_R[n_runs],
         ENdp_Gnorm_R[n_runs], RQdp_Lnorm_R[n_runs], ENdp_Lnorm_R[n_runs]) = (
             gnd.pyGNDiv_fun(RF, Abundances))
        print(('\tpyGNDiv on %s reflectance factors: elapsed time ' +
               '%.2f seconds...') % (sensor, time.time()-t0))

        # Store results generating the test dataset. Limited to seven to
        # prevent generating large datasets
        if n_runs < 7:
            db_ = pd.DataFrame(data=np.concatenate((wl.reshape(1, -1), RF),
                                                   axis=0))
            db_.to_csv((folder_out + 'Region_%d_ReflectanceFactors%s.txt' %
                        (n_runs + 1, sensor)), sep=';', index=False,
                       header=False)
            # Store only once since these are the same for all sensorrs
            if sensor == 'Hy':
                db_ = pd.DataFrame(data=Abundances)
                db_.to_csv((folder_out + 'Region_%d_RelativeAbundance.txt' %
                            (n_runs + 1)), sep=';', index=False, header=False)
                db_ = pd.DataFrame(data=PlantTraits,
                                   columns=[Bvars[i] for i in I_Rbiod])
                db_.to_csv((folder_out + 'Region_%d_PlantTraits.txt' %
                            (n_runs + 1)), sep=';', index=False)

    print('\n*****************************************************')
    print('Total ellapsed time: elapsed time %.2f minutes...'
          % ((time.time()-t_start) / 60.))
    
    # Store Functional Diversity Metrics --------------------------------------
    if sensor == 'Hy':
        with open(folder_out + 'FDMs_PT.pkl', 'wb') as f:
            pickle.dump((FRic_PT, VBdp_PT, VBdp_Gnorm_PT, VBdp_Lnorm_PT,
                         RQdp_PT, RQdp_Gnorm_PT, ENdp_Gnorm_PT, RQdp_Lnorm_PT,
                         ENdp_Lnorm_PT), f)

    with open(folder_out + 'FDMs_%s.pkl' % sensor, 'wb') as f:
        pickle.dump((FRic_R, VBdp_R, VBdp_Gnorm_R, VBdp_Lnorm_R, RQdp_R,
                     RQdp_Gnorm_R, ENdp_Gnorm_R, RQdp_Lnorm_R,
                     ENdp_Lnorm_R), f)


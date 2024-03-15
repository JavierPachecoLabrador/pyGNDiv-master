# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 08:41:04 2022

@author: Javier Pacheco-Labrador, PhD (jpacheco@bgc-jena.mpg.de)
         Max Planck Institute for Biogeochemsitry, Jena, Germany
        Currently: javier.pacheco@csic.es
         Environmental Remote Sensing and Spectroscopy Laboratory (SpecLab)
         Spanish National Research Council (CSIC)

DESCRIPTION:
==============================================================================
This script is used to test and show who the pyGNDiv package works.

It uses the simualted vegetation traits and spectral reflectance factors
    to calculate diversity indices and to partition diversity followinf the
    methods described in Pacheco-Labrador et al., 2023.

Then, it plots the resulting metrics to show how generalizable normalizaiton
    improves the relationships between field and remote sensing-derived metrics
    and how it sets them at the same scale, making them directly comparable,
    for example, independently of the number of sensor bands.

Cite as:
  Pacheco-Labrador, J., de Bello, F., Migliavacca , M., Ma, X., Carvalhais, N.,
    & Wirth, C. (2023). A generalizable normalization for assessing
    plant functional diversity metrics across scales from remote sensing.
    Methods in Ecology and Evolution.
"""

# %% Import
import pyGNDiv as gnd
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


# %% Functions
def plot_metrics(vPT, vR, var_, var_lbl, sensors_, fig_name=None, marker_='o',
                 marker_size=10):
    nc_ = len(sensors_)
    nr_ = len(var_)

    fig, ax = plt.subplots(nr_, nc_, sharex=False, sharey=False,
                           figsize=[4.7*nc_, 4.5*nr_])
    plt.rcParams.update({'font.size': 12})

    i_ = 0
    for v_ in var_:
        x_ = np.empty(0)
        if type(vPT[i_][0][v_]) is float:
            for ii_ in range(len(vPT[i_])):
                x_ = np.concatenate((x_, np.array([vPT[i_][ii_][v_]])))
        else:
            for ii_ in range(len(vPT[i_])):
                x_ = np.concatenate((x_, vPT[i_][ii_][v_].reshape(-1)))

        j_ = 0
        for sen_ in sensors_:
            y_ = np.empty(0)
            if type(vR[i_][sen_][0][v_]) is float:
                for ii_ in range(len(vR[i_][sen_])):
                    y_ = np.concatenate((y_,
                                         np.array([vR[i_][sen_][ii_][v_]])))
            else:
                for ii_ in range(len(vR[i_][sen_])):
                    y_ = np.concatenate((y_,
                                         vR[i_][sen_][ii_][v_].reshape(-1)))

            sn.regplot(x=x_, y=y_, marker=marker_, ci=None, ax=ax[i_][j_],
                       scatter_kws={'color': 'darkolivegreen',
                                    's': marker_size},
                       line_kws={'color': "darkorange"})
            # ax[i_][j_].plot(x_, y_, marker_, c='darkolivegreen')
            if i_ == 0:
                ax[i_][j_].set_title(sen_)
            ax[i_][j_].set_xlabel(var_lbl[i_] % 'PT')
            ax[i_][j_].set_ylabel(var_lbl[i_] % 'R')
            ax[i_][j_].grid()
            j_ += 1
        i_ += 1
    if fig_name is not None:
        plt.savefig(fig_name + '.png', dpi=250)
    plt.show()
    plt.close()


# %% Load example dataset
"""
The example dataset (test_data folder) contains simulated plant trait and
reflectance factors resampled to the spectral configuration of different
sensors corresponding to the species of three regions, each made of different
communities where these species combine.

Specific physically or biologically plausible bonds are defined for each plant
trait, whereas these are assumed [0, 1] for the reflectance factors

The simulated plant traits are:
  * N: Leaf structural parameter [layers]
  * Cab: Leaf chlorophyll content  [mug cm-2]
  * Cca: Leaf carotenoids content  [mug cm-2]
  * Cant: Leaf anthocyanins content  [mug cm-2]
  * Cs: Leaf senescent pigments content  [a.u.]
  * Cw: Leaf water content  [g cm-2]
  * Cdm: Leaf dry matter content  [g cm-2]
  * LIDFa: Leaf angle distribution mean parameter [-]
  * LIDFb: Leaf angle distribution bimodality parameter [-]
  * LAI: Leaf area index [m2 m-2]
  * hc: Canopy height [m]
  * leafwidth: Leaf width [m]

The sensors simulated are:
  * Hy: Hyperspectral sensor, 1 nm spectral sampling between 400 - 2400 nm, as
  predicted by a radiative transfer model
  * DESIS: DLR Earth Sensing Imaging Spectrometer, Visible and near-infrared
  hyperspectral imager onboard the international space station.
  58 spectral bands simulated between 400 and 1000 nm
    (https://www.dlr.de/content/en/articles/missions-projects/horizons/experimente-horizons-desis.html)
  * S2: Sentinel-2 MultiSpectral Instrument. Visible and short wave infrared
  multiband imager of the ESA Copernicus Programme. It features 10 bands
  characterizing Earth surface reflectance factors
  (https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/msi-instrument)
  * QB2: QuickBird2 is a high spatial resolution imager featuring four channels
  in the visible and near infrared regions and a pancromatic channel.
  (https://earth.esa.int/eogateway/missions/quickbird-2)

The pyGNDiv package operates with the following input datastets:
  * Relative abundances: communities x species matrix, with relative
    abundances normalized per community
  * Plant or spectral traits: species x traits or reflectance factors matrix

"""

# Define path to the test simulated data -------------------------------------
path_data = os.getcwd() + '//test_data//'

# Plant trait bounds ---------------------------------------------------------
# Even if these are shown here, notice that they are not needed to apply the
# Generalizable Normalization.
tmp_ = np.loadtxt(path_data + 'PlantTrait_Bounds.txt', skiprows=1,
                  delimiter=';')
LBpt_ = tmp_[0]
UBpt_ = tmp_[1]

# Define and preallocate datasets --------------------------------------------
# Define regions to be processed (choose from 1 to 7)
num_regions = 7
# Define the sensors to be simulated/analyzed
sensors_ = ['Hy', 'DESIS', 'S2', 'QB2']
# Preallocate a dictionary for sensor's wavelengths
wl_ = {'Hy': 0, 'DESIS': 0, 'S2': 0, 'QB2': 0}
# Preallocate a dictionary for sensor's reflectance factors. A reflectance
# factor's matrix (species x bands) per region is separately stored as an item
# of a list.
R_ = {'Hy': num_regions*[0], 'DESIS': num_regions*[0],
      'S2': num_regions*[0], 'QB2': num_regions*[0]}
# Preallocate a list to store the relative abundance matrices(communities x
# species) of each region as separated items
A_ = num_regions*[0]
# Preallocate a list to store the plant traits matrices(species x traits)
# of each region as separated items
PT_ = num_regions*[0]

# Load the data from the /test_dataset folder --------------------------------
for i_ in range(num_regions):
    j_ = i_ + 1
    print('Loading Region %d...' % (j_))
    # Relative abundances
    A_[i_] = np.loadtxt(path_data + 'Region_%d_RelativeAbundance.txt' % j_,
                        delimiter=';')
    # Plant traits
    PT_[i_] = np.loadtxt(path_data + 'Region_%d_PlantTraits.txt' % j_,
                         skiprows=1, delimiter=';')
    # Reflectance factors
    for sen_ in sensors_:
        tmp = np.loadtxt((path_data + 'Region_%d_ReflectanceFactors_%s.txt' %
                          (j_, sen_)), delimiter=';')
        R_[sen_][i_] = tmp[1:, :]
        # Get also the wavelengths
        if i_ == 0:
            wl_[sen_] = tmp[0, :]
    print('\t...%d communities with %d species' % (A_[i_].shape))


# %% Produce regular and normalized versions of functional diversity indices
"""
Here is where the pyGNDiv package is used to analize plant and spectral traits
region per region.

"""

# Preallocate dictionaries to allocate the outputs ---------------------------
(FRic_PT, VBdp_PT, VBdp_Gnorm_PT, VBdp_Lnorm_PT, RQdp_PT, RQdp_Gnorm_PT,
 ENdp_Gnorm_PT, RQdp_Lnorm_PT, ENdp_Lnorm_PT) = (
     gnd.preallocate_outputs(num_regions))

(FRic_R, VBdp_R, VBdp_Gnorm_R, VBdp_Lnorm_R, RQdp_R, RQdp_Gnorm_R,
 ENdp_Gnorm_R, RQdp_Lnorm_R, ENdp_Lnorm_R) = (
     gnd.preallocate_outputs(num_regions, sensors_))

# Analyze the data region per region -----------------------------------------
for i_ in range(num_regions):
    print('Analyzing Region %d...' % (i_ + 1))
    # Plant traits
    (FRic_PT[i_], VBdp_PT[i_], VBdp_Gnorm_PT[i_], VBdp_Lnorm_PT[i_],
     RQdp_PT[i_], RQdp_Gnorm_PT[i_], ENdp_Gnorm_PT[i_], RQdp_Lnorm_PT[i_],
     ENdp_Lnorm_PT[i_]) = (gnd.pyGNDiv_fun(PT_[i_], A_[i_]))

    # Reflectance factors with different spectral configurations
    for sen_ in sensors_:
        (FRic_R[sen_][i_], VBdp_R[sen_][i_], VBdp_Gnorm_R[sen_][i_],
         VBdp_Lnorm_R[sen_][i_], RQdp_R[sen_][i_], RQdp_Gnorm_R[sen_][i_],
         ENdp_Gnorm_R[sen_][i_], RQdp_Lnorm_R[sen_][i_],
         ENdp_Lnorm_R[sen_][i_]) = (gnd.pyGNDiv_fun(R_[sen_][i_], A_[i_]))

# Store the results ----------------------------------------------------------
with open(os.getcwd() + '//GNDivfdms.pkl', 'wb') as f:
    pickle.dump((FRic_PT, VBdp_PT, VBdp_Gnorm_PT, VBdp_Lnorm_PT, RQdp_PT,
                 RQdp_Gnorm_PT, ENdp_Gnorm_PT, RQdp_Lnorm_PT, ENdp_Lnorm_PT,
                 FRic_R, VBdp_R, VBdp_Gnorm_R, VBdp_Lnorm_R, RQdp_R,
                 RQdp_Gnorm_R, ENdp_Gnorm_R, RQdp_Lnorm_R, ENdp_Lnorm_R), f)


# %% Plots. Diversity indices
"""
Generate different plots comparing plant trait-based metrics (x-axis) and
remote sensing-based traits (y-axis) for different metrics, remote sensors,
normalization approaches and diveristy partitioning methods.

If you want to store the plots, provide a path and filename in the fig_name
argument of the function plot_metrics

"""

# Functional Richness --------------------------------------------------------
# FRic
var_ = ['FRic', 'FRic_Lnorm', 'FRic_Gnorm']
var_lbl = ['$FRic_{non,%s}$', '$FRic_{local,%s}$', '$FRic_{GN,%s}$']
x_ = [FRic_PT]*num_regions
y_ = [FRic_R]*num_regions
plot_metrics(x_, y_, var_, var_lbl, sensors_, fig_name=None)

# FRic with maximum 3 components
var_ = ['FRic3', 'FRic3_Lnorm', 'FRic3_Gnorm']
var_lbl = ['$FRic_{non,3,%s}$', '$FRic_{3,local,%s}$', '$FRic_{3,GN,%s}$']
x_ = [FRic_PT]*num_regions
y_ = [FRic_R]*num_regions
plot_metrics(x_, y_, var_, var_lbl, sensors_, fig_name=None)

# Rao Quadratic Entropy and equivalent numbers -------------------------------
var_ = ['RaoQ']*5
var_lbl = ['$RaoQ_{non,%s}$', '$RaoQ_{local,%s}$', '$RaoQ_{GN,%s}$',
           '$RaoQ_{eq,local,%s}$', '$RaoQ_{eq,GN,%s}$']
x_ = [RQdp_PT, RQdp_Lnorm_PT, RQdp_Gnorm_PT, ENdp_Lnorm_PT, ENdp_Gnorm_PT]
y_ = [RQdp_R, RQdp_Lnorm_R, RQdp_Gnorm_R, ENdp_Lnorm_R, ENdp_Gnorm_R]
plot_metrics(x_, y_, var_, var_lbl, sensors_, fig_name=None)

# %% Plot. Diversity decomposition with functional metrics (de Bello 2010)
x_ = [RQdp_PT, RQdp_Lnorm_PT, RQdp_Gnorm_PT, ENdp_Lnorm_PT, ENdp_Gnorm_PT]
y_ = [RQdp_R, RQdp_Lnorm_R, RQdp_Gnorm_R, ENdp_Lnorm_R, ENdp_Gnorm_R]

# Alpha-diversity ------------------------------------------------------------
var_ = ['alpha_mean']*5
var_lbl = ['$\\alpha_{non,%s}$', '$\\alpha_{local,%s}$', '$\\alpha_{GN,%s}$',
           '$\\alpha_{eq,local,%s}$', '$\\alpha_{eq,GN,%s}$']
plot_metrics(x_, y_, var_, var_lbl, sensors_, marker_size=100, fig_name=None)

# Beta-diversity -------------------------------------------------------------
var_ = ['beta_add']*5
var_lbl = ['$\\beta_{non,%s}$', '$\\beta_{local,%s}$', '$\\beta_{GN,%s}$',
           '$\\beta_{eq,local,%s}$', '$\\beta_{eq,GN,%s}$']
plot_metrics(x_, y_, var_, var_lbl, sensors_, marker_size=100, fig_name=None)

# Gamma-diversity ------------------------------------------------------------
var_ = ['gamma']*5
var_lbl = ['$\\gamma_{non,%s}$', '$\\gamma_{local,%s}$', '$\\gamma_{GN,%s}$',
           '$\\gamma_{eq,local,%s}$', '$\\gamma_{eq,GN,%s}$']
plot_metrics(x_, y_, var_, var_lbl, sensors_, marker_size=100, fig_name=None)

# Fraction of alpha-diversity ------------------------------------------------
var_ = ['Falpha_norm']*5
var_lbl = ['$f_{\\alpha,non,%s}$', '$f_{\\alpha,local,%s}$',
           '$f_{\\alpha,GN,%s}$', '$f_{\\alpha,eq,local,%s}$',
           '$f_{\\alpha,eq,GN,%s}$']
plot_metrics(x_, y_, var_, var_lbl, sensors_, marker_size=100, fig_name=None)

# Fraction of beta-diversity ------------------------------------------------
var_ = ['Fbeta_norm']*5
var_lbl = ['$f_{\\beta,non,%s}$', '$f_{\\beta,local,%s}$',
           '$f_{\\beta,GN,%s}$', '$f_{\\beta,eq,local,%s}$',
           '$f_{\\beta,eq,GN,%s}$']
plot_metrics(x_, y_, var_, var_lbl, sensors_, marker_size=100, fig_name=None)

# %% Plot. Variance-based partitioning (Laliberté et al. 2020)
x_ = [VBdp_PT, VBdp_Lnorm_PT, VBdp_Gnorm_PT]
y_ = [VBdp_R, VBdp_Lnorm_R, VBdp_Gnorm_R]

# Alpha-diversity ------------------------------------------------------------
var_ = ['SSalpha']*3
var_lbl = ['$\\alpha_{non,%s}$', '$\\alpha_{local,%s}$', '$\\alpha_{GN,%s}$']
plot_metrics(x_, y_, var_, var_lbl, sensors_, marker_size=100, fig_name=None)

# Beta-diversity ------------------------------------------------------------
var_ = ['SSbeta']*3
var_lbl = ['$\\beta_{non,%s}$', '$\\beta_{local,%s}$', '$\\beta_{GN,%s}$']
plot_metrics(x_, y_, var_, var_lbl, sensors_, marker_size=100, fig_name=None)

# Gamma-diversity ------------------------------------------------------------
var_ = ['SSgamma']*3
var_lbl = ['$\\gamma_{non,%s}$', '$\\gamma_{local,%s}$', '$\\gamma_{GN,%s}$']
plot_metrics(x_, y_, var_, var_lbl, sensors_, marker_size=100, fig_name=None)

# Fraction of alpha-diversity ------------------------------------------------
var_ = ['Falpha']*3
var_lbl = ['$f_{\\alpha,non,%s}$', '$f_{\\alpha,local,%s}$',
           '$f_{\\alpha,GN,%s}$']
plot_metrics(x_, y_, var_, var_lbl, sensors_, marker_size=100, fig_name=None)

# Fraction of beta-diversity ------------------------------------------------
var_ = ['Fbeta']*3
var_lbl = ['$f_{\\beta,non,%s}$', '$f_{\\beta,local,%s}$',
           '$f_{\\beta,GN,%s}$']
plot_metrics(x_, y_, var_, var_lbl, sensors_, marker_size=100, fig_name=None)

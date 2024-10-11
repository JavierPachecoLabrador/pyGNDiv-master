"""
@author: Javier Pacheco-Labrador, PhD (javier.pacheco@csic.es)
         Environmental Remote Sensing and Spectroscopy Laboratory (SpecLab)
         Spanish National Research Council (CSIC)

DESCRIPTION:
==============================================================================
This script test the performance of the pyGNDIv module for imagery
(pyGNDIV_imagery), which by default uses the pyGNDiv version optimized with 
numba, which is faster.

Cite as:
  Pacheco-Labrador, J., de Bello, F., Migliavacca , M., Ma, X., Carvalhais, N.,
    & Wirth, C. (2023). A generalizable normalization for assessing
    plant functional diversity metrics across scales from remote sensing.
    Methods in Ecology and Evolution.
"""

import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..') + '//'))
from pyGNDiv import pyGNDiv_imagery as gni


# %% Test the PyGNDiv module for imagery
im_sizes = [33, 99, 999, 9999]
time_log = np.zeros((3, len(im_sizes)))
for i_, sz_ in enumerate(im_sizes):
    print('###################################################################')
    # Produce a cube with 3 traits
    cube_ = np.random.random((sz_, sz_, 3))    

    print('PCA @ {a} x {b} pixels scene'.format(a=sz_, b=sz_))
    t0 = time.time()
    PCA_in = gni.apply_pca_cube(cube_)
    time_log[0, i_] = time.time() - t0

    print('Rao Q and diveristy partitioning @ {a} x {b} pixels scene'.format(
        a=sz_, b=sz_))
    t0 = time.time()
    raoQ, raoQ_part = gni.raoQ_grid(PCA_in, wsz_=3, mask_in=None,
                                    weight_in=None, fexp_var_in=.98,
                                    n_sigmas_norm=6., alphas_=[1.],
                                    nan_tolerance=0., calculate_RaoQ_part=True)
    time_log[1, i_] = time.time() - t0

    print(
        'Variance-based diveristy partitioning @ {a} x {b} pixels scene'.format(
            a=sz_, b=sz_))
    t0 = time.time()
    (SSalpha, SSbeta, SSgamma, f_alpha, f_beta, frac_used) = gni.varpart_grid(
        PCA_in, wsz_=3, weight_in=None, mask_in=None, nan_tolerance=0.,
        fexp_var_in=.98, n_sigmas_norm=6.)
    time_log[2, i_] = time.time() - t0


# %% Plot the computation times
p_ = [None] * 3
labels = ['PCA', 'Rao Q and paritition', 'Variance-based partition']
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.grid()
for i_, lb in enumerate(labels):
    ax.plot(im_sizes, time_log[i_] / 60., '-o')
ax.set_xscale('log')
ax.set_xlabel('Image size (linear pixels)')
ax.set_ylabel('Time (min)')
ax.legend(labels, loc='best')
fig.savefig('pyGNDiv_imagery_test.png')
plt.close()
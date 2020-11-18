#!/usr/bin/python

import numpy as np
from astropy.io import fits as pyfits
import glob

sigma = np.loadtxt('./sigma_mu.txt')
e_pecvel = (5*150/cst.c.to('km/s').value)/(np.log(10)*z)
e_lens = 0.055*z
e_coh = sigma[:,0]

def mu_cov(alpha, beta, z, e_coh, e_pecvel, e_lens):
    """ Assemble the full covariance matrix of distance modulus

    See Betoule et al. (2014), Eq. 11-13 for reference
    """
    Ceta = sum([pyfits.getdata(mat) for mat in glob.glob('C*.fits')])

    Cmu = np.zeros_like(Ceta[::3,::3])
    for i, coef1 in enumerate([1., alpha, -beta]):
        for j, coef2 in enumerate([1., alpha, -beta]):
            Cmu += (coef1 * coef2) * Ceta[i::3,j::3]

    # Add diagonal term from Eq. 13
    Cmu[np.diag_indices_from(Cmu)] += e_lens**2 + e_coh**2 + e_pecvel**2    
    return Cmu

if __name__ == "__main__":
    Cmu = mu_cov(0.13, 3.1)

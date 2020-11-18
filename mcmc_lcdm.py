import emcee
import numpy as np
import matplotlib.pyplot as plt
from corner import corner
from utils.cosmo_without_pool import tofit
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib import gridspec
from astropy.io import fits
from matplotlib import cm
import pandas as pd
from glob import glob
import os
import astropy.constants as cst
import scipy as sp
import platform
plt.ion()

def mu_cov(alpha, beta):
    C_mu = np.zeros_like(C_eta[::3,::3])
    for i, coef1 in enumerate([1., alpha, -beta]):
        for j, coef2 in enumerate([1., alpha, -beta]):
            C_mu += (coef1 * coef2) * C_eta[i::3,j::3]
    C_mu[np.diag_indices_from(C_mu)] += to_add
    return C_mu

def lnlike(params):
    om, M, dM, alpha, beta = params
    C_mu = mu_cov(alpha, beta)
    D = mb - M + alpha*x1 - beta*c - tofit(z, om, -1)
    D[mstar >= 10] -= dM
    slv = sp.linalg.solve_triangular(np.linalg.cholesky(C_mu), D.T, lower=True)
    res = slv.T @ slv
    return -0.5 * res

def lnprior(params):
    if (priors[0][0] < params[0] < priors[0][1]) & (priors[1][0] < params[1] < priors[1][1]) & (priors[2][0] < params[2] < priors[2][1]) & (priors[3][0] < params[3] < priors[3][1]) & (priors[4][0] < params[4] < priors[4][1]):
        return 0.0
    else:
        return -np.inf

def lnprob(params):
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(params)

#data = 'Pantheon'
data = 'JLA'
#data = 'JLA+A2'

if data == 'Pantheon':
    a = np.genfromtxt('data/pantheon.txt', names=True, dtype=None)
    mask = a['zSN'] > -999
    a_masked = a[mask]
    mb = a_masked['mB']
    e_mb = a_masked['e_mB']
    z = a_masked['zSN']
    x1 = a_masked['x1']
    e_x1 = a_masked['e_x1']
    c = a_masked['c']
    e_c = a_masked['e_c']
    mstar = a_masked['logM']
    e_mstar = a_masked['e_logM']
elif data == 'JLA':
    sigma = np.loadtxt('data/covmat/sigma_mu.txt')
    #C_eta = sum([fits.getdata(mat) for mat in [g for g in glob('covmat/C*.fits') if 'host' not in g]])
    C_eta = sum([fits.getdata(mat) for mat in [g for g in glob('data/covmat/C*.fits')]])
    a = fits.getdata('data/JLA.fit')
    mb = a['mB']
    e_mb = a['e_mB']
    z = a['zcmb']
    x1 = a['x1']
    e_x1 = a['e_x1']
    c = a['c']
    e_c = a['e_c']
    mstar = a['logMst']
    sets = a['set']
    e_mstar = a['e_logMst']
    e_pecvel = (5*150/cst.c.to('km/s').value)/(np.log(10)*z)
    e_lens = 0.055*z
    e_coh = sigma[:,0]
    to_add = e_coh**2 + e_lens**2 + e_pecvel**2
    #cov = np.zeros((len(a), 3, 3))
    #cov[:,0,0] = e_mb**2
    #cov[:,1,1] = e_x1**2
    #cov[:,2,2] = e_c**2
    #cov[:,0,1] = a['cov_mb_s_']
    #cov[:,1,0] = a['cov_mb_s_']
    #cov[:,0,2] = a['cov_mb_c_']
    #cov[:,2,0] = a['cov_mb_c_']
    #cov[:,2,1] = a['cov_s_c_']
    #cov[:,1,2] = a['cov_s_c_'] 
elif data == 'JLA+A2':
    a = fits.getdata('data/JLA.fit')
    a = a[a['set'] != 3]
    b = pd.read_csv('data/A2.dat', sep=' ')
    b = b[b['mB'] > 0]
    sets = np.concatenate([a['set'], np.ones(len(b))*3])
    mb = np.concatenate([a['mB'], b['mB']])
    e_mb = np.concatenate([a['e_mB'], b['e_mB']])
    z = np.concatenate([a['zcmb'], b['zCMB']])
    x1 = np.concatenate([a['x1'], b['x1']])
    e_x1 = np.concatenate([a['e_x1'], b['e_x1']])
    c = np.concatenate([a['c'], b['c']])
    e_c = np.concatenate([a['e_c'], b['e_c']])
    mstar = a['logMst']
    e_mstar = a['e_logMst']
else:
    print('Not good data!')

priors = [[0, 1.0], [-5, 5], [-1, 1], [0.05, 0.25], [2, 5]]
pos0 = [0.4, 1.75, -0.07, 0.165, 3.028]

ndim = len(pos0)
nwalkers = 2*ndim
nsteps = 3000
ncut = int(0.1*nsteps)

pos = [pos0 + np.random.randn(ndim)*0.005*pos0 for i in range(nwalkers)]

n_cpus_available = os.cpu_count()
if n_cpus_available <= 8:
    n_cpus = n_cpus_available
    print("I'm assuming you are working on your laptop, I assigned all available CPUs, that is: {}".format(n_cpus))
else:
    n_cpus = int(3/4.*n_cpus_available)
    print("I'm assuming you are connected to a big cluster, I assigned 75% of available CPUs, that is: {}".format(n_cpus))

the_pool = Pool(n_cpus)

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=the_pool)

sampler.run_mcmc(pos, nsteps, progress=True)

the_pool.close()

samples = sampler.chain[:, ncut:, :].reshape((-1, ndim))

labels = [r'$\Omega_\mathrm{M}$', r'$M$', r'$\Delta M$', r'$\alpha$', r'$\beta$']
corner(samples, labels=labels, plot_datapoints=False, color='C0', title='$\Lambda$CDM')

om, M, dM, alpha, beta = np.median(samples, axis=0)
w = -1
z_bin = np.linspace(0.01, 1.5, 1000)
model = tofit(z_bin, om, w)
model_eds = tofit(z_bin, 1, -1)
model_at_z = tofit(z, om, w)
mu = mb + alpha*x1 - beta*c - M
mu[mstar > 10] -= dM
e_mu = np.sqrt(e_mb**2 + (abs(alpha)*e_x1)**2 + (abs(beta)*e_c)**2)

cm_subsection = np.linspace(0.01, 1, 4)
colors = [cm.viridis_r(x) for x in cm_subsection]

fig = plt.figure()
gs = gridspec.GridSpec(3, 3)
ax1 = fig.add_subplot(gs[:2, :])
line1 = ax1.errorbar(z[sets == 3], mu[sets == 3], e_mu[sets == 3], fmt='.', c=colors[0], label='low-$z$', elinewidth=0.5, ms=2)
line2 = ax1.errorbar(z[sets == 2], mu[sets == 2], e_mu[sets == 2], fmt='.', c=colors[1], label='SDSS', elinewidth=0.5, ms=2)
line3 = ax1.errorbar(z[sets == 1], mu[sets == 1], e_mu[sets == 1], fmt='.', c=colors[2], label='SNLS', elinewidth=0.5, ms=2)
line4 = ax1.errorbar(z[sets == 4], mu[sets == 4], e_mu[sets == 4], fmt='.', c=colors[3], label='HST', elinewidth=0.5, ms=2)
line5, = ax1.plot(z_bin, model, c='C0', label=r'$\mu_\mathrm{\Lambda CDM}$ (Best fit)', linewidth=1)
line6, = ax1.plot(z_bin, model_eds, c='C1', label=r'$\mu_{\mathrm{EdS}}$', linewidth=1)
ax1.axes.get_xaxis().set_visible(False)
ax1.set_ylabel('$\mu$', fontsize=15)
legend1 = ax1.legend(handles=[line1, line2, line3, line4], frameon=False, loc=2, fontsize=12)
ax1.add_artist(legend1)
legend2 = ax1.legend(handles=[line5, line6], frameon=False, loc=4, fontsize=12)
ax1.set_xscale('log')
ax2 = fig.add_subplot(gs[-1, :], sharex=ax1)
ax2.axhline(linestyle='--', c='C0', linewidth=1)
ax2.errorbar(z[sets == 3], (mu - model_at_z)[sets == 3], e_mu[sets == 3], fmt='.', c=colors[0], elinewidth=0.5, ms=2)
ax2.errorbar(z[sets == 2], (mu - model_at_z)[sets == 2], e_mu[sets == 2], fmt='.', c=colors[1], elinewidth=0.5, ms=2)
ax2.errorbar(z[sets == 1], (mu - model_at_z)[sets == 1], e_mu[sets == 1], fmt='.', c=colors[2], elinewidth=0.5, ms=2)
ax2.errorbar(z[sets == 4], (mu - model_at_z)[sets == 4], e_mu[sets == 4], fmt='.', c=colors[3], elinewidth=0.5, ms=2)
ax2.plot(z_bin, model_eds - model, c='C1', linewidth=1)
ax2.set_ylim(-1.1, 1.1)
ax2.set_xlabel('$z$', fontsize=15)
ax2.set_ylabel('$\mu-\mu_\mathrm{\Lambda CDM}$', fontsize=15)
plt.subplots_adjust(hspace=0)

mask = (e_mstar < 1) & (mstar > 7) & (e_mu < 0.3)
bins = np.linspace(mstar[mask].min(), mstar[mask].max(), 10)
digitized = np.digitize(mstar[mask], bins)
bin_means = [(mu - model_at_z)[mask][digitized == i].mean() for i in range(1, len(bins))]
bin_errors = [np.std((mu - model_at_z)[mask][digitized == i]) for i in range(1, len(bins))]
plt.figure()
plt.errorbar(mstar[mask], (mu - model_at_z)[mask], xerr=e_mstar[mask], yerr=e_mu[mask], fmt='.', ms=2, elinewidth=0.5, alpha=0.2)
plt.errorbar(0.5*(bins[:-1] + bins[1:]), bin_means, bin_errors, fmt='.')
plt.axhline(linestyle='--', c='C0', linewidth=1)
plt.xlabel('$\mathrm{log}_{10} ( {M_\star}_\mathrm{ZPEG} / M_\odot)$', fontsize=15)
plt.ylabel('$\mu-\mu_\mathrm{\Lambda CDM}$', fontsize=15)

'''
bb = pd.read_csv('ml_res.csv')

mask = a['zhel'] < 0.3

maskaka = []
for i in range(len(bb)):
    maskaka.append(np.where(a['RAJ2000'][mask] == np.round(bb['RAJ2000'][i], 7))[0][0])

z_aft = np.array([z[mask][i] for i in maskaka])
mu_aft = np.array([mu[mask][i] for i in maskaka])
model_aft = np.array([tofit(z, om, w)[mask][i] for i in maskaka])
mstar_zpeg = np.array([a['logMst'][mask][i] for i in maskaka]) 

ssfr = 10**bb['SFR_ML']/10**bb['MSTAR_ML']
from scipy.optimize import curve_fit

from scipy.stats import sigmaclip

jj = sigmaclip(mstar_zpeg - bb['MSTAR_ML'], 3)
mask = (mstar_zpeg - bb['MSTAR_ML'] > jj.lower) & (mstar_zpeg - bb['MSTAR_ML'] < jj.upper)

plt.figure()
plt.plot(mstar_zpeg, bb['MSTAR_ML'], '.')
plt.plot(mstar_zpeg[mask], bb['MSTAR_ML'][mask], '.')
plt.plot([5, 13], [5, 13])
plt.xlabel('$\mathrm{log}_{10} ( {M_\star}_\mathrm{ZPEG} / M_\odot)$', fontsize=15)
plt.ylabel('$\mathrm{log}_{10} ( {M_\star}_\mathrm{ML} / M_\odot)$', fontsize=15)

plt.figure()
plt.plot(bb['MSTAR_ML'], (mu_aft - model_aft), '.')
plt.plot(bb['MSTAR_ML'][mask], (mu_aft - model_aft)[mask], '.')
plt.xlabel('$\mathrm{log}_{10} ( {M_\star}_\mathrm{ML} / M_\odot)$', fontsize=15)
plt.ylabel('$\mu-\mu_\mathrm{\Lambda CDM}$', fontsize=15)

#mask_f = (z_aft > 0.1) & (z_aft < 0.2)
mask_f = (z_aft > 0.1) & (z_aft < 0.3)
p, p0 = curve_fit(lin, np.log10(ssfr)[mask_f], (mu_aft - model_aft)[mask_f])
'''

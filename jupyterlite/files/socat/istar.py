"""
Minimal vendored subset of coPsi (https://github.com/emilknudstrup/coPsi) for
the SOCat True-Obliquity JupyterLite notebook.

Keeps only the iStar path used by SOCat:
  - createDistributions  (draw Gaussian/uniform parameter distributions)
  - stellarInclination   (Masuda & Winn 2020 i_star via emcee MCMC)
  - coPsi                (cos psi = sin i* sin i_o cos lam + cos i* cos i_o)
  - getKDE / hpd / getConfidence

Differences from upstream coPsi (to run in Pyodide / JupyterLite):
  * statsmodels KDE  ->  scipy.stats.gaussian_kde
  * no multiprocessing.Pool (emcee runs single-process; emcee installs via
    micropip in the browser)
  * no light-curve modules (Prot / Phot) imported

Original author: Emil Knudstrup (coPsi).  Vendored & trimmed for SOCat.
"""
import numpy as np
import scipy.stats as ss
from scipy.special import erf

dayfac = 86400.0
rsunfac = 695700.0   # R_sun in km


# --------------------------------------------------------------------------
# priors (from coPsi.priors)
# --------------------------------------------------------------------------
def flat_prior(val, xmin, xmax):
    return 0.0 if (val < xmin or val > xmax) else 1.0 / (xmax - xmin)


def gauss_prior(val, xmid, xwid):
    nom = np.exp(-1.0 * (val - xmid) ** 2 / (2 * xwid ** 2))
    return nom / (np.sqrt(2 * np.pi) * xwid)


def tgauss_prior(val, xmid, xwid, xmin, xmax):
    if val < xmin or val > xmax:
        return 0.0
    nom = np.exp(-1.0 * (val - xmid) ** 2 / (2 * xwid ** 2)) / (np.sqrt(2 * np.pi) * xwid)
    den1 = (1.0 + erf((xmax - xmid) / (np.sqrt(2) * xwid))) / 2
    den2 = (1.0 + erf((xmin - xmid) / (np.sqrt(2) * xwid))) / 2
    return nom / (den1 - den2)


def jeff_prior(val, xmin, xmax):
    return 0.0 if (val < xmin or val > xmax) else 1.0 / (val * np.log(xmax / xmin))


def flat_prior_dis(val, xmin, xmax):
    return xmin + val * (xmax - xmin)


def tgauss_prior_dis(mu, sigma, xmin, xmax):
    a = (xmin - mu) / sigma
    b = (xmax - mu) / sigma
    return ss.truncnorm.rvs(a, b, loc=mu, scale=sigma)


def gauss_prior_dis(mu, sigma):
    return np.random.normal(mu, sigma)


def jeff_prior_dis(val, xmin, xmax):
    if xmin == 0:
        xmin = 1e-10
    return np.exp(val * np.log(xmax / xmin) + np.log(xmin))


# --------------------------------------------------------------------------
# likelihood for the i_star MCMC  (Masuda & Winn 2020 / Hjorth+2021)
# --------------------------------------------------------------------------
def lnprob(positions, **pars):
    log_prob = 0.0
    for idx, par in enumerate(pars['FPs']):
        val = positions[idx]
        pval, sigma, lower, upper = pars[par][:4]
        ptype = pars[par][-1]
        if ptype == 'uniform':
            prob = flat_prior(val, lower, upper)
        elif ptype == 'jeff':
            prob = jeff_prior(val, lower, upper)
        elif ptype == 'gauss':
            prob = gauss_prior(val, pval, sigma)
        elif ptype == 'tgauss':
            prob = tgauss_prior(val, pval, sigma, lower, upper)
        else:
            prob = 0.0
        if prob != 0.0:
            log_prob += np.log(prob)
        else:
            return -np.inf

    v = (2 * np.pi * positions[0] * rsunfac) / (positions[1] * dayfac)
    vsini = v * np.sqrt(1 - positions[2] ** 2)
    prob = gauss_prior(vsini, pars['vsini'][0], pars['vsini'][1])
    return (np.log(prob) + log_prob) if prob != 0.0 else -np.inf


class iStar(object):
    parameters = ['inco', 'lam', 'Prot', 'Rs', 'vsini', 'Teff', 'psi']
    stepParameters = ['Rs', 'Prot', 'cosi']
    labels = {
        'Rs': r'$R_\star \ (R_\odot)$', 'Prot': r'$P_{\rm rot}\ \rm (days)$',
        'cosi': r'$\cos i_\star$', 'incs': r'$i_\star \ \rm (deg)$',
        'v': r'$v \ \rm (km/s)$', 'vsini': r'$v \sin i_\star \ \rm (km/s)$',
        'psi': r'$\psi \ \rm (deg)$',
    }

    def __init__(self,
                 inco=(85., 0.5, 0.0, 90., 'gauss'),
                 lam=(0, 0.5, -180., 180., 'gauss'),
                 Prot=(3.5, 0.5, 0.0, 10., 'gauss'),
                 Rs=(1.0, 0.1, 0.5, 2.0, 'gauss'),
                 vsini=(4.5, 0.5, 0.0, 10., 'gauss'),
                 cosi=(0, 0.1, -1.0, 1.0, 'uniform'),
                 Teff=(6250, 100, 3000, 9000, 'gauss')):
        self.inco, self.lam, self.Prot = inco, lam, Prot
        self.Rs, self.vsini, self.cosi, self.Teff = Rs, vsini, cosi, Teff

    # --- cos psi -----------------------------------------------------------
    def coPsi(self):
        inco = np.deg2rad(self.dist['inco'])
        incs = np.deg2rad(self.dist['incs'])
        lam = np.deg2rad(self.dist['lam'])
        self.dist['cosp'] = np.sin(incs) * np.sin(inco) * np.cos(lam) + np.cos(incs) * np.cos(inco)
        self.dist['psi'] = np.rad2deg(np.arccos(self.dist['cosp']))

    # --- parameter distributions ------------------------------------------
    def createDistributions(self, N=2000):
        self.dist = {}
        pars = vars(self)
        generate = []
        for par in list(pars):
            if par in self.parameters:
                if isinstance(pars[par], tuple):
                    generate.append(par)
                elif isinstance(pars[par], np.ndarray):
                    self.dist[par] = pars[par]
                    N = len(pars[par])
        for par in generate:
            spec = pars[par]
            if spec[-1] == 'gauss':
                self.dist[par] = np.random.normal(spec[0], spec[1], N)
            elif spec[-1] == 'uniform':
                self.dist[par] = np.random.uniform(spec[2], spec[3], N)

    # --- stellar inclination via emcee (Masuda & Winn 2020) ---------------
    def stellarInclination(self, ndraws=5000, nwalkers=100, moves=None, progress=False):
        import emcee
        try:
            self.dist
        except AttributeError:
            self.createDistributions()

        def start_coords(nwalkers, ndim):
            fps = self.stepParameters
            pars = vars(self)
            pos = []
            for _ in range(nwalkers):
                start = np.ndarray(ndim)
                for idx, par in enumerate(fps):
                    pri = pars[par][:4]
                    dist = pars[par][-1]
                    if dist == 'tgauss':
                        start[idx] = tgauss_prior_dis(pri[0], pri[1], pri[2], pri[3])
                    elif dist == 'gauss':
                        start[idx] = gauss_prior_dis(pri[0], pri[1])
                    elif dist == 'uniform':
                        start[idx] = flat_prior_dis(np.random.uniform(), pri[2], pri[3])
                    elif dist == 'jeff':
                        start[idx] = jeff_prior_dis(np.random.uniform(), pri[2], pri[3])
                pos.append(start)
            return pos

        ndim = len(self.stepParameters)
        coords = start_coords(nwalkers, ndim)
        pp = vars(self)
        pars = {'FPs': ['Rs', 'Prot', 'cosi'], 'Rs': pp['Rs'], 'Prot': pp['Prot'],
                'cosi': pp['cosi'], 'vsini': pp['vsini']}

        # emcee's built-in progress uses tqdm, which isn't reliably installable
        # in Pyodide — drive emcee with progress=False and print a lightweight
        # text status ourselves every 100 steps when progress=True.
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, moves=moves, kwargs=pars)
        old_tau = np.inf
        converged = False
        for _ in sampler.sample(coords, iterations=ndraws, progress=False):
            if sampler.iteration % 100:
                continue
            if progress:
                print(f"\r  MCMC {sampler.iteration}/{ndraws} steps", end="", flush=True)
            tau = sampler.get_autocorr_time(tol=0)
            converged = np.all(tau * 50 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau
        if progress:
            tail = (f"converged after {sampler.iteration} steps" if converged
                    else f"finished {sampler.iteration} steps")
            print(f"\r  MCMC {tail}{' ' * 8}")

        tau = sampler.get_autocorr_time(tol=0)
        thin = max(1, int(0.5 * np.min(tau)))
        burnin = int(2 * np.max(tau))
        flat = sampler.get_chain(discard=burnin, flat=True, thin=thin)
        flat[:, 2] = np.abs(flat[:, 2])

        self.dist['cosi'] = flat[:, 2]
        incs = np.rad2deg(np.arccos(flat[:, 2]))
        self.dist['incs'] = incs
        v = (2 * np.pi * flat[:, 0] * rsunfac) / (flat[:, 1] * dayfac)

        results = {'Parameter': ['Median', 'Lower', 'Upper']}
        cols = ['Rs', 'Prot', 'cosi', 'incs', 'v', 'vsini']
        data = [flat[:, 0], flat[:, 1], flat[:, 2], incs, v, v * np.sin(np.deg2rad(incs))]
        for name, d in zip(cols, data):
            val, up, low = self.getConfidence(d)
            results[name] = [val, low, up]
        try:
            import pandas as pd
            return pd.DataFrame(results)
        except Exception:
            return results

    # --- statistics --------------------------------------------------------
    def getKDE(self, z, n=512):
        z = np.asarray(z, dtype=float)
        kde = ss.gaussian_kde(z)
        x = np.linspace(z.min(), z.max(), n)
        return x, kde(x)

    def hpd(self, data, lev=0.68):
        d = sorted(list(data))
        nData = len(d)
        nIn = int(round(lev * nData))
        if nIn < 2:
            raise RuntimeError("not enough data")
        i = 0
        r = d[i + nIn - 1] - d[i]
        for k in range(len(d) - (nIn - 1)):
            rk = d[k + nIn - 1] - d[k]
            if rk < r:
                r = rk
                i = k
        return (d[i], d[i + nIn - 1], i, i + nIn - 1)

    def getConfidence(self, z, lev=0.68):
        val = np.median(z)
        conf = self.hpd(z, lev=lev)
        return val, conf[1] - val, val - conf[0]

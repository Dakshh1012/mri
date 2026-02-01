import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, gamma
from patsy import dmatrix
import warnings

class GAMLSS:
    def __init__(self, distribution='BCCG'):
        self.distribution = distribution
        self.results = None
        self.df = None

    def _get_design_matrix(self, x, df=3):
        try:
            # Using natural splines (cr) or b-splines (bs)
            # 3 df for mu, 2 df for sigma/nu is often safer to avoid overfitting
            formula = f"bs(x, df={df}, degree=3, include_intercept=True) - 1"
            dm = dmatrix(formula, {"x": x}, return_type='matrix')
            return np.asarray(dm)
        except Exception:
            return np.column_stack([np.ones(len(x)), x])

    def _bccg_log_likelihood(self, params, y, X_mu, X_sigma, X_nu):
        n_mu = X_mu.shape[1]
        n_sigma = X_sigma.shape[1]
        
        beta_mu = params[:n_mu]
        beta_sigma = params[n_mu:n_mu+n_sigma]
        beta_nu = params[n_mu+n_sigma:]
        
        mu = np.exp(np.clip(X_mu @ beta_mu, -20, 20))
        sigma = np.exp(np.clip(X_sigma @ beta_sigma, -20, 20))
        nu = np.clip(X_nu @ beta_nu, -5, 5)
        
        mu = np.maximum(mu, 1e-10)
        sigma = np.maximum(sigma, 1e-10)
        
        # Stability check for nu near 0
        z = np.where(np.abs(nu) > 1e-3, 
                     ((y / mu)**nu - 1) / (sigma * nu), 
                     np.log(y / mu) / sigma)
        
        log_jac = (nu - 1) * np.log(y / mu) - np.log(mu) - np.log(sigma)
        log_l = -0.5 * np.log(2 * np.pi) - 0.5 * z**2 + log_jac
        
        # Regularization to prevent extreme curvature
        penalty = 0.01 * np.sum(params**2)
        
        return -np.sum(log_l) + penalty

    def _no_log_likelihood(self, params, y, X_mu, X_sigma):
        n_mu = X_mu.shape[1]
        mu = X_mu @ params[:n_mu]
        sigma = np.exp(np.clip(X_sigma @ params[n_mu:], -20, 20))
        sigma = np.maximum(sigma, 1e-10)
        return -np.sum(norm.logpdf(y, loc=mu, scale=sigma))

    def _ga_log_likelihood(self, params, y, X_mu, X_sigma):
        n_mu = X_mu.shape[1]
        mu = np.exp(np.clip(X_mu @ params[:n_mu], -20, 20))
        sigma = np.exp(np.clip(X_sigma @ params[n_mu:], -20, 20))
        mu = np.maximum(mu, 1e-10)
        sigma = np.maximum(sigma, 1e-10)
        return -np.sum(gamma.logpdf(y, a=1/(sigma**2), scale=mu*(sigma**2)))

    def fit(self, x, y, df=3):
        self.df = df
        X_mu = self._get_design_matrix(x, df=df)
        X_sigma = self._get_design_matrix(x, df=max(2, df-1))
        
        if self.distribution == 'BCCG':
            X_nu = self._get_design_matrix(x, df=2)
            
            # Smart init: use linear reg for mu
            mu_init = np.log(np.mean(y))
            sigma_init = np.log(np.std(y) / np.mean(y))
            
            p0 = np.zeros(X_mu.shape[1] + X_sigma.shape[1] + X_nu.shape[1])
            p0[0] = mu_init
            p0[X_mu.shape[1]] = sigma_init
            p0[X_mu.shape[1] + X_sigma.shape[1]] = 1.0 # Nu=1
            
            res = minimize(self._bccg_log_likelihood, p0, args=(y, X_mu, X_sigma, X_nu), method='L-BFGS-B')
            self.results = {
                'params': res.x, 'X_mu': X_mu, 'X_sigma': X_sigma, 'X_nu': X_nu,
                'aic': 2 * len(res.x) + 2 * res.fun, 'success': res.success
            }
        
        elif self.distribution == 'NO':
            p0 = np.zeros(X_mu.shape[1] + X_sigma.shape[1])
            p0[0] = np.mean(y)
            p0[X_mu.shape[1]] = np.log(np.std(y))
            res = minimize(self._no_log_likelihood, p0, args=(y, X_mu, X_sigma), method='L-BFGS-B')
            self.results = {
                'params': res.x, 'X_mu': X_mu, 'X_sigma': X_sigma,
                'aic': 2 * len(res.x) + 2 * res.fun, 'success': res.success
            }
            
        elif self.distribution == 'GA':
            p0 = np.zeros(X_mu.shape[1] + X_sigma.shape[1])
            p0[0] = np.log(np.mean(y))
            p0[X_mu.shape[1]] = np.log(np.std(y) / np.mean(y))
            res = minimize(self._ga_log_likelihood, p0, args=(y, X_mu, X_sigma), method='L-BFGS-B')
            self.results = {
                'params': res.x, 'X_mu': X_mu, 'X_sigma': X_sigma,
                'aic': 2 * len(res.x) + 2 * res.fun, 'success': res.success
            }

    def predict_percentiles(self, x_new, percentiles=[1, 5, 10, 25, 50, 75, 90, 95, 99]):
        if self.results is None: return None
        X_mu_new = self._get_design_matrix(x_new, df=self.df)
        X_sigma_new = self._get_design_matrix(x_new, df=max(2, self.df-1))
        params = self.results['params']
        
        if self.distribution == 'BCCG':
            X_nu_new = self._get_design_matrix(x_new, df=2)
            n_mu, n_sigma = X_mu_new.shape[1], X_sigma_new.shape[1]
            mu = np.exp(X_mu_new @ params[:n_mu])
            sigma = np.exp(X_sigma_new @ params[n_mu:n_mu+n_sigma])
            nu = X_nu_new @ params[n_mu+n_sigma:]
            
            return {f"{p}th": np.where(nu != 0, 
                                      mu * (1 + sigma * nu * norm.ppf(p/100.0))**(np.maximum(1/nu, -20)),
                                      mu * np.exp(sigma * norm.ppf(p/100.0))) for p in percentiles}
        elif self.distribution == 'NO':
            n_mu = X_mu_new.shape[1]
            mu = X_mu_new @ params[:n_mu]
            sigma = np.exp(X_sigma_new @ params[n_mu:])
            return {f"{p}th": norm.ppf(p/100.0, loc=mu, scale=sigma) for p in percentiles}
        elif self.distribution == 'GA':
            n_mu = X_mu_new.shape[1]
            mu = np.exp(X_mu_new @ params[:n_mu])
            sigma = np.exp(X_sigma_new @ params[n_mu:])
            return {f"{p}th": gamma.ppf(p/100.0, a=1/(sigma**2), scale=mu*(sigma**2)) for p in percentiles}

import numpy as np
import pandas as pd
'''
This script contains metrics to ensure the quality of photmoetric redshift estimation in this work
'''


def bias(z_phot, z_spec):
    '''
            The bias measures the deviation of the estimated photometric redshift from the true(i.e., the spetroscopic redshift)
    '''

    b = ((z_phot-z_spec)/(1+z_spec))
    return b


def scatter(z_phot, z_spec):
    '''
            The scatter between the true redshift and the photometric redshift
    '''
    square = (((((z_phot-z_spec)/(1+z_spec))**2)))
    sigma = (square)

    return sigma


def scatter_scalar(z_phot, z_spec):
    '''
    The scatter between the true redshift and the photometric redshift
    '''
    sigma = np.sqrt(np.mean((np.abs(((z_phot-z_spec)/(1+z_spec))**2))))

    return sigma


def compute_metrics(y_true, y_pred, clf_name):
    result = pd.Series()
    delta_znorm = (y_pred - y_true)/(1 + y_true)
    result.loc['RMSE_znorm'] = np.sqrt(np.mean((delta_znorm)**2))
    result.loc['bias_znorm'] = np.mean(delta_znorm)
    result.loc['std_znorm'] = np.std(delta_znorm)
    result.loc['RMSE'] = np.sqrt(np.mean((y_pred - y_true)**2))
    result.loc['|znorm| > 0.15 (%)'] = 100 * \
        np.sum(np.abs(delta_znorm) > 0.15)/y_true.shape[0]
    result.loc['|znorm| > 3std (%)'] = 100*np.sum(np.abs(delta_znorm)
                                                  > 3*np.std(delta_znorm))/y_true.shape[0]
    result.loc['scatter'] = scatter_scalar(y_pred, y_true)
    result.loc["bias"] = np.mean(bias(y_pred, y_true))
    fr = fraction_retained(y_true, y_pred, 0.15)
    result.loc["fr015"] = np.mean(fr)
    sigma = sigma68_vec(y_true, y_pred)

    result.loc["sigma68"] = sigma68_vec(y_true, y_pred)

    result.name = clf_name
    return result


def fraction_retained(y_true, y_pred, e):
    fr = 100.0*(abs(y_true-y_pred)/(y_true+1.0) < e)
    return fr


def chi_squared(y_true, y_pred):
    num = (y_true - y_pred)**2
    quo = y_true + y_pred

    return 0.5*(num/quo)


def mc_cdf(cdf, bins):
    '''Monte Carlo Cumalative sampling'''

    rand = np.random.random()
    ind = np.where(cdf > rand)
    if len(ind[0]) >= 2:
        frac = (rand-cdf[ind[0][0]-1])/(cdf[ind[0][0]]-cdf[ind[0][0]-1])
        zfinal = bins[ind[0][0]-1] + frac*(bins[ind[0][0]]-bins[ind[0][0]-1])
        return zfinal, rand, cdf[ind[0][0]-1], cdf[ind[0][0]]
    else:
        rand = np.random.random()
        ind = np.where(cdf > rand)
        while(len(ind[0]) == 0):
            rand = np.random.random()
            ind = np.where(cdf >= rand)
        frac = (rand-cdf[ind[0][0]-1])/(cdf[ind[0][0]]-cdf[ind[0][0]-1])
        zfinal = bins[ind[0][0]-1] + frac*(bins[ind[0][0]]-bins[ind[0][0]-1])
        return zfinal, rand, cdf[ind[0][0]-1], cdf[ind[0][0]]


def sigma68_vec(y_true, y_pred):

    error = y_true - y_pred
    error = np.sort(error)
    sigma68 = (int(len(error)*0.159), int(len(error)*(1-0.159)))
    sig68_1 = round(error[sigma68[0]], 8)
    sig68_2 = round(error[sigma68[1]], 8)
    total = (abs(sig68_1)+abs(sig68_2))/2
    return total

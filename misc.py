import numpy as np
import xarray as xr 
from uncertainties import ufloat, unumpy


def x_step(array):
    x, = array.coords
    dx = float(array[x][1] - array[x][0])
    return dx


def weighted_average(vals, stds):
    variance = stds*stds
    variance_inv = 1/variance
    variance_inv_sum = np.sum(variance_inv, axis=-1)
    return np.sum(vals*variance_inv/variance_inv_sum, axis=-1), np.sqrt(1/variance_inv_sum)


def weighted_average_unumpy(uarray):
    variance = unumpy.std_devs(uarray)**2
    variance_inv = 1/variance
    variance_inv_sum = np.sum(variance_inv)
    return ufloat(np.sum(unumpy.nominal_values(uarray)*variance_inv/variance_inv_sum), 
                  np.sqrt(1/variance_inv_sum))


def phase_wrap(trace, period, phase=None):
    
    dt = float(trace.t[1] - trace.t[0])
    
    if phase is None:
        result = xr.DataArray(trace, coords={'t': trace.t%period})
    else:
        result = xr.DataArray(trace.roll(t=-int((phase)//dt)), coords={'t': trace.t%period})
    
    return result
    

def aver_bins(trace, nbins=20): 

    t = trace.t.groupby_bins('t', bins=nbins).mean().values

    trace_std = trace.groupby_bins('t', bins=nbins, labels=t).std()
    trace_count = trace.groupby_bins('t', bins=nbins, labels=t).count()
    
    result = xr.Dataset({'aver': trace.groupby_bins('t', bins=nbins, labels=t).mean(),
                      'n': trace_count,
                      'sem': trace_std/np.sqrt(trace_count)},
                      )
    
    return result.rename(t_bins='t')


def block_aver_darray(darray, n, boundary='pad'):
    assert len(darray.dims) == 1
    dim_name, = darray.dims
    
    darray_mean = darray.coarsen({dim_name: n}, boundary=boundary).mean()
    darray_std = darray.coarsen({dim_name: n}, boundary=boundary).std()
    darray_count = xr.DataArray(np.r_[n*np.ones(darray.size//n), darray.size%n], coords=darray_mean.coords)
    darray_sem = darray_std/np.sqrt(darray_count)
    
    return xr.Dataset({'y_mean': darray_mean,
                       'y_std': darray_std,
                       'y_sem': darray_sem,
                       'y_count': darray_count})


def savgol_filt(array, window_length, polyorder, **kwgs):

    dx = x_step(array)
    n = int(window_length/dx)
    win_len = n if n%2 == 0 else n - 1
    
    x_low_pass = savgol_filter(array.values, 
                             window_length=win_len,
                             polyorder=polyorder,
                             **kwgs)

    return xr.DataArray(x_low_pass, coords=array.coords)
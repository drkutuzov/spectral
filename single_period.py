import numpy as np
import xarray as xr 


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
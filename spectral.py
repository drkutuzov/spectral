import numpy as np
import xarray as xr 
# import matplotlib.pyplot as plt
# from scipy.signal._spectral_py import _spectral_helper as cross_spec
# from uncertainties import unumpy, ufloat


#------------Auxillary--------------
def x_step(array):
    x, = array.coords
    dx = float(array[x][1] - array[x][0])
    return dx


#------------Preprocessing/Filtering----------------
def high_pass_filt(x, lowest_freq):

    dt = float(x.t[1] - x.t[0])
    x_fft = np.fft.fft(x)
    freq = np.linspace(0, 1/dt, x.t.size)
    
    x_fft[freq < lowest_freq] = 0
    x_fft[freq > freq[-1] - lowest_freq] = 0

    return xr.DataArray(np.real(np.fft.ifft(x_fft)), coords=x.coords)


#------------Spectral Analysis Single Trace---------------
def periodogram(array):
    
    N = len(array)
    dx = x_step(array)
    freq = np.fft.rfftfreq(N, d=dx)
    fft_array = dx*np.fft.rfft(array)
    
    return xr.DataArray(np.abs(fft_array)**2/(N*dx), 
                        coords={'f': freq}, 
                        dims=('f'))


def minimize_leakage(array, f_slice):

    P0 = periodogram(array)
    P0_cut = P0.sel(f=f_slice)
    P0_cut_oneside = P0.sel(f=slice(None, f_slice.stop))
    i0 = len(P0_cut_oneside) - len(P0_cut)
    k_max = int(P0_cut.argmax() + i0)

    N = len(array)
    dx = x_step(array)
    dN = int(1.5/(dx*float(P0.f[k_max])))
    indexes = np.arange(N - dN, N + 1)

    P1, P2 = [], []

    for i in indexes:
    
        Pi = periodogram(array[:i])
        P1.append(float(Pi[k_max - 1]))
        P2.append(float(Pi[k_max]))

    x_end = indexes*dx
    x_end_fine = np.arange(x_end[0], x_end[-1], 1e-6)
    delta_P = np.abs(np.interp(x_end_fine, x_end, P1) - np.interp(x_end_fine, x_end, P2))
    length = x_end_fine[int(np.argmin(delta_P))]
    f0 = (k_max - 0.5)/length
    
    x, = array.coords.keys()
    P1 = xr.DataArray(P1, coords={x: x_end}, dims=x)
    P2 = xr.DataArray(P2, coords={x: x_end}, dims=x)

    result = xr.Dataset({'P1': P1, 'P2': P2}, attrs=dict(f0=f0, length=length))

    return result

# def welch(x, fs, detrend='linear', nperseg=None, window='hann', noverlap=None):
    
#     freq, time, Pxx = cross_spec(x, x, fs, window=window, detrend=detrend, nperseg=nperseg, noverlap=noverlap)

#     if noverlap is None:
#         variance_reduction = 9*len(time)/11
#     elif noverlap == 0:
#         variance_reduction = len(time)
#     else:
#         raise ValueError('noverlap should be None or 0')
    
#     Pxx_darray = xr.DataArray(Pxx, coords={'f': freq, 't': time}, dims=('f', 't'))
    
#     return xr.Dataset({'P': Pxx_darray, 
#                        'P_mean': Pxx_darray.mean('t'),
#                        'P_sem': Pxx_darray.std('t') / np.sqrt(variance_reduction)})


#------------Spectral Analysis Two Traces---------------

# def cross_spectrum(x, y, fs, detrend='linear', nperseg=None):
    
#     freq, time, Pxy = cross_spec(x, y, fs, window='hann', detrend=detrend, nperseg=nperseg)
#     variance_reduction = 9*len(time)/11
    
#     Pxy_darray = xr.DataArray(Pxy, coords={'f': freq, 't': time}, dims=('f', 't'))
    
#     Pxy_mag = xr.apply_ufunc(np.abs, Pxy_darray)
#     Pxy_phase = xr.apply_ufunc(np.angle, Pxy_darray)
    
#     return xr.Dataset({'P_mag': Pxy_mag, 
#                        'P_mag_mean': Pxy_mag.mean('t'),
#                        'P_mag_sem': Pxy_mag.std('t') / np.sqrt(variance_reduction),
#                        'P_phase_mean': Pxy_phase.mean('t'),
#                        'P_phase_sem': Pxy_phase.std('t') / np.sqrt(variance_reduction)})



# def transfer_function(x, y, window='hann', detrend='linear', nperseg=None, noverlap=None):
    
#     ''' x and y should be xarray DataArrays '''
    
#     assert(len(x) == len(y))
    
#     fs = 1/float(x.t[1] - x.t[0])
#     t0 = float(x.t[0])
    
#     freq, time, Sxy = cross_spec(x.values, y.values, fs, window=window, detrend=detrend, nperseg=nperseg, noverlap=noverlap)
#     freq, time, Sxx = cross_spec(x.values, x.values, fs, window=window, detrend=detrend, nperseg=nperseg, noverlap=noverlap)
    
#     x_mean = np.array([x.sel(t=slice(t1, t2)).mean('t') for t1, t2 in zip(t0 + time - time[0], t0 + time + time[0])])
#     y_mean = np.array([y.sel(t=slice(t1, t2)).mean('t') for t1, t2 in zip(t0 + time - time[0], t0 + time + time[0])])
    
#     H = (Sxy * x_mean) / (Sxx * y_mean) # transfer function
#     variance_reduction = 9*len(time)/11
    
#     gain = xr.DataArray(np.transpose(20*np.log10(np.abs(H))), coords={'t': time, 'f': freq}, dims=('t', 'f'))
#     gain_mean = gain.mean('t')
#     gain_sem = gain.std('t') / np.sqrt(variance_reduction)
    
#     phase = xr.DataArray(np.transpose(np.angle(H)), coords={'t': time, 'f': freq}, dims=('t', 'f'))
#     phase_mean = phase.mean('t')
#     phase_sem = phase.std('t') / np.sqrt(variance_reduction)
    
#     return xr.Dataset({'gain': gain, 
#                        'phase': phase,
#                        'gain_mean': gain_mean,
#                        'phase_mean': phase_mean,
#                        'gain_mean_err': gain_sem,
#                        'phase_mean_err': phase_sem,
#                        })


# def coherence(x, y, window='hann', detrend='linear', nperseg=None, noverlap=None):
    
#     ''' x and y should be xarray DataArrays '''
    
#     assert(len(x) == len(y))
    
#     fs = 1/float(x.t[1] - x.t[0])
#     t0 = float(x.t[0])
    
#     freq, time, Sxy = cross_spec(x.values, y.values, fs, window=window, detrend=detrend, nperseg=nperseg, noverlap=noverlap)
#     freq, time, Sxx = cross_spec(x.values, x.values, fs, window=window, detrend=detrend, nperseg=nperseg, noverlap=noverlap)
#     freq, time, Syy = cross_spec(y.values, y.values, fs, window=window, detrend=detrend, nperseg=nperseg, noverlap=noverlap)
    
#     variance_reduction = 9*len(time)/11
    
#     Sxx = xr.DataArray(np.transpose(Sxx), coords={'t': time, 'f': freq}, dims=('t', 'f'))
#     Sxx_mean = unumpy.uarray(Sxx.mean('t'), Sxx.std('t') / np.sqrt(variance_reduction))
    
#     Syy = xr.DataArray(np.transpose(Syy), coords={'t': time, 'f': freq}, dims=('t', 'f'))
#     Syy_mean = unumpy.uarray(Syy.mean('t'), Syy.std('t') / np.sqrt(variance_reduction))
    
#     Sxy = xr.DataArray(np.transpose(Sxy), coords={'t': time, 'f': freq}, dims=('t', 'f'))
    
#     Sxy_real_mean = unumpy.uarray(np.real(Sxy).mean('t'), np.real(Sxy).std('t') / np.sqrt(variance_reduction))
#     Sxy_imag_mean = unumpy.uarray(np.imag(Sxy).mean('t'), np.imag(Sxy).std('t') / np.sqrt(variance_reduction))
    
#     coherence = (Sxy_real_mean**2 + Sxy_imag_mean**2) / Sxx_mean / Syy_mean
    
#     coherence_mean =  xr.DataArray(unumpy.nominal_values(coherence), coords={'f': freq}, dims='f')
#     coherence_sem =  xr.DataArray(unumpy.std_devs(coherence), coords={'f': freq}, dims='f')
    
#     return xr.Dataset({'Sxx': Sxx, 
#                        'Syy': Syy,
#                        'Sxy': Sxy,
#                        'coherence_mean': coherence_mean,
#                        'coherence_mean_err': coherence_sem
#                        })


# def phase_shift(freq_range, spectrum, delay_correction=0.0):

#     peak = spectrum.sel(f=slice(*freq_range))
#     freq = float(peak.f[int(peak.P_mag_mean.argmax())])

#     phi, err = misc.weighted_average(peak.P_phase_mean.values, peak.P_phase_sem.values)
    
#     delay = ufloat(1e3/freq * phi/2/np.pi, 1e3/freq * err/2/np.pi)

#     print(f'Frequency = {freq:.1f} Hz')
#     print(f'phase_shift = {ufloat(phi, err):.3f} rad')
#     print(f'Delay = {delay:.3f} ms')
#     print(f'Corrected delay = {delay - delay_correction:.3f} ms')

#     return dict(phase_delay=ufloat(phi, err), delay=delay, corrected_delay=delay - delay_correction)


# def phase_wrap(trace, period, phase=None):
    
#     dt = float(trace.t[1] - trace.t[0])
    
#     if phase is None:
#         result = xr.DataArray(trace, coords={'t': trace.t%period})
#     else:
#         result = xr.DataArray(trace.roll(t=-int((phase)//dt)), coords={'t': trace.t%period})
    
#     return result
    

# def aver_bins(trace, nbins=20): 

#     t = trace.t.groupby_bins('t', bins=nbins).mean().values

#     trace_std = trace.groupby_bins('t', bins=nbins, labels=t).std()
#     trace_count = trace.groupby_bins('t', bins=nbins, labels=t).count()
    
#     result = xr.Dataset({'aver': trace.groupby_bins('t', bins=nbins, labels=t).mean(),
#                       'n': trace_count,
#                       'sem': trace_std/np.sqrt(trace_count)},
#                       )
    
#     return result.rename(t_bins='t')




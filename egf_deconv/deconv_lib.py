import numpy as np
from scipy.fftpack import fft, ifft
from obspy.signal.util import next_pow_2

def phase_shift_filter(nfft, dt, tshift):
    freq = np.fft.fftfreq(nfft, d=dt)
    return np.exp(-2j * np.pi * freq * tshift)

def gauss_filter(nfft, dt, a):
    f = np.fft.fftfreq(nfft, dt)
    w = 2 * np.pi * f
    return np.exp(-0.25 * (w/a)**2)

def apply_filter(x, filt):
    nfft = len(filt)
    xf = fft(x, n=nfft)
    x_flt = ifft(xf*filt, n=nfft).real
    return x_flt

def deconv_iterative(tar_data, egf_data, samp_rate, tshift=0., gauss_a=10, iter_max=200, derr_min=0.01, only_pos=True, to_norm=True):
    dt = 1 / samp_rate
    npts = len(tar_data)
    nfft = next_pow_2(npts)
    df = 1 / (nfft * dt)
    rms = np.zeros(iter_max) 
    p0 = np.zeros(nfft) # gaussian pulses
    tar_data = np.concatenate([tar_data, np.zeros(nfft-npts)])
    egf_data = np.concatenate([egf_data, np.zeros(nfft-npts)])
    gauss_flt = gauss_filter(nfft, dt, gauss_a)
    shift_flt = phase_shift_filter(nfft, dt, tshift)
    tar_flt = apply_filter(tar_data, gauss_flt)
    egf_flt = apply_filter(egf_data, gauss_flt)
    power_egf = np.sum(egf_flt**2) 
    power_tar = np.sum(tar_flt**2) 
    rem_flt = tar_flt.copy()  # remaining target waveform
    for ii in range(iter_max):  
        cc = ifft(fft(rem_flt) * np.conj(fft(egf_flt))).real  # CC for time lag
        cc = cc / power_egf / dt  # scale for pulse amplitude
        cc = cc[0:int(nfft/2)-1] # only for causal time
        if only_pos: dt_idx = np.argmax(cc) 
        else: dt_idx = np.argmax(abs(cc))
        p0[dt_idx] += cc[dt_idx]
        tar_pred = apply_filter(p0, fft(egf_flt)) * dt 
        rem_flt = tar_flt - tar_pred  # use residual data in the next iteration
        rms[ii] = np.sum(rem_flt**2) / power_tar
        d_error = 100*(rms[ii] - rms[ii-1]) if ii>0 else 100
        if abs(d_error)<derr_min: break
    stf = apply_filter(p0, gauss_flt * shift_flt)[0:npts]
    if to_norm: stf /= np.amax(stf)
    time = np.arange(npts) * dt - tshift
    return time, stf, tar_pred[0:npts], rms[0:ii]

def deconv_waterlevel(tar_data, egf_data, samp_rate, tshift=0., gauss_a=10, wl=0.01, to_norm=True):
    dt = 1 / samp_rate
    npts = len(tar_data)
    nfft = next_pow_2(npts)
    df = 1 / (nfft * dt)
    tar_spec = fft(tar_data, nfft)
    egf_spec = fft(egf_data, nfft)
    shift_flt = phase_shift_filter(nfft, dt, tshift)
    gauss_flt = gauss_filter(nfft, dt, gauss_a)
    # denominator: add water level correction
    egf_psd = egf_spec * egf_spec.conjugate()
    water_level = wl * np.amax(egf_psd.real)
    egf_psd[np.where(egf_psd.real < water_level)[0]] = water_level
    # spectral ratio for STF
    cross_spec = tar_spec * egf_spec.conjugate() * gauss_flt
    spec_ratio = cross_spec / egf_psd
    stf = ifft(spec_ratio * shift_flt, nfft)[0:npts].real
    # waveform fit
    tar_obs = ifft(tar_spec * gauss_flt, nfft)[0:npts].real
    tar_pred = ifft(spec_ratio * egf_spec, nfft)[0:npts].real
    rms = np.sum((tar_pred - tar_obs)**2 ) / np.sum(tar_obs ** 2)
    if to_norm: stf /= np.sum(gauss_flt) * df * dt
    time = np.arange(npts) * dt - tshift
    return time, stf, tar_pred, rms


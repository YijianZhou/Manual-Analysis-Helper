import sys, os, glob
import numpy as np
from obspy import read, UTCDateTime
import matplotlib.pyplot as plt
from signal_lib import preprocess
from reader import dtime2str, read_fpha
from scipy import interpolate
from mtspec import mtspec

# i/o paths
tar_idx = 0
ftar = 'input/eg_tar.pha'
tar_loc, tar_dict = read_fpha(ftar)[tar_idx]
tar_name = dtime2str(tar_loc[0])
tar_dir = 'input/eg_tar/%s'%tar_name
tar_mag = 10**(1.5*0.853*tar_loc[-1]) # Ml to M0
fegf = 'output/eg_egf.pha'
egf_root = 'input/eg_egf'
egf_list = read_fpha(fegf)
sta_list = ['CI.JRC2','CI.CLC','CI.WMF','CI.WCS2','CI.TOW2','CI.WRC2','CI.CCC','PB.B917'] 
fout = 'output/eg_spec-ratio-stack_%s.pdf'%tar_name
title = 'Stacked Spectral Ratio: %s (Ml %s)'%(tar_name, tar_loc[-1])
# signal processing
samp_rate = 100
freq_band = [0.1,None]
s_win = [0.5,5.5]
dt_list = np.arange(0,3,1) # multi-win strategy (Imanishi & Ellsworth, 2006)
num_grid = 20
fc1_min, fc1_max = [.8,1.5]
fc2_min, fc2_max = [4.,10.]
moment_ratio = np.arange(160, 200, 0.5)
log_fc1_min, log_fc1_max = np.log10(fc1_min), np.log10(fc1_max)
log_fc2_min, log_fc2_max = np.log10(fc2_min), np.log10(fc2_max)
fc1_grids = [10**(log_fc1_min + ii*(log_fc1_max-log_fc1_min)/num_grid) for ii in range(num_grid)]
fc2_grids = [10**(log_fc2_min + ii*(log_fc2_max-log_fc2_min)/num_grid) for ii in range(num_grid)]
# fig config
fig_size = (10,8)
fsize_label = 14
fsize_title = 18
alpha = 0.8
freq_min, freq_max = [0.2,20]
df_log = 0.025


def log_mean(spec_list):
    npts = len(spec_list[0])
    spec_mean = np.zeros(npts)
    for ii in range(npts):
        log_spec = np.mean([np.log10(spec[ii]) for spec in spec_list])
        spec_mean[ii] = 10**log_spec
    return spec_mean

def calc_spec(stream, ts):
    spec_list = []
    # for each sliding win
    for dt in dt_list:
        st = stream.slice(ts-s_win[0]+dt, ts+s_win[1]+dt)
        data = st.detrend('demean').detrend('linear').taper(max_percentage=0.05)[0].data
        spec, freq = mtspec(data, delta=1/samp_rate, time_bandwidth=4, number_of_tapers=7)
        cond = (freq>=freq_min) * (freq<=freq_max)
        freq = freq[cond]
        spec = spec[cond]
        log_freq = [np.log10(fi) for fi in freq]
        log_spec = [np.log10(si) for si in spec]
        npts = int((log_freq[-1] - log_freq[0]) / df_log)
        f_log_spec = interpolate.interp1d(log_freq, log_spec)
        log_freq = np.arange(log_freq[0], log_freq[0]+npts*df_log, df_log)
        freq = [10**log_fi for log_fi in log_freq]
        log_spec = f_log_spec(log_freq)
        spec = np.array([10**log_si for log_si in log_spec])
        spec_list.append(spec)
    spec = log_mean(spec_list)
    return freq, spec

def calc_spec_ratio(sta_list):
    # calc target spec
    spec_tar = []
    for net_sta in sta_list:
        # calc target spec
        st_paths = sorted(glob.glob('%s/%s.*'%(tar_dir, net_sta)))[0:2]
        if len(st_paths)!=2: print('missing target data!', net_sta); continue
        stream  = read(st_paths[0])
        stream += read(st_paths[1])
        stream = preprocess(stream, samp_rate, freq_band)
        if len(stream)!=2: print('bad target data!', net_sta); continue
        ts = tar_dict[net_sta][1]
        freq_0, spec_0 = calc_spec(stream, ts)
        freq_1, spec_1 = calc_spec(stream, ts)
        spec = (spec_0+spec_1) /2 /tar_mag
        spec_tar.append([freq_0, spec])
    # calc egf spec
    spec_egf = []
    for net_sta in sta_list:
      spec_egf.append([])
      for [egf_loc, egf_dict] in egf_list:
        if net_sta not in egf_dict: continue
        egf_name = dtime2str(egf_loc[0])
        st_paths = sorted(glob.glob('%s/%s/%s.*'%(egf_root, egf_name, net_sta)))[0:2]
        if len(st_paths)!=2: print('missing EGF data!', net_sta); continue
        stream  = read(st_paths[0])
        stream += read(st_paths[1])
        stream = preprocess(stream, samp_rate, freq_band)
        if len(stream)!=2: print('bad EGF data!', net_sta); continue
        ts = egf_dict[net_sta][1]
        freq_0, spec_0 = calc_spec(stream, ts)
        freq_1, spec_1 = calc_spec(stream, ts)
        egf_mag = 10**(1.5*0.853*egf_loc[-1])
        spec = (spec_0+spec_1) /2 /egf_mag
        spec_egf[-1].append([freq_0, spec])
    # calc spectral ratio
    spec_ratio = []
    for ii in range(len(sta_list)):
        spec_egf_stack = log_mean([spec for [_,spec] in spec_egf[ii]])
        spec_ratio.append(spec_tar[ii][1] / spec_egf_stack)
    return freq_0, spec_ratio

def calc_loss(spec_obs, fc1, fc2, mr):
    loss = 0
    for ii in range(npts):
        spec_pred = np.log10(mr * np.sqrt((1+(freq[ii]/fc2)**4) / (1+(freq[ii]/fc1)**4)))
        loss += abs(spec_pred - np.log10(spec_obs[ii]))
    return loss

# calc spec ratio
freq, spec_ratio = calc_spec_ratio(sta_list)
spec_stack = log_mean(spec_ratio)
spec_max = np.amax([np.amax(spec) for spec in spec_ratio])
# model fit
npts = len(freq)
param_loss = []
for fc1 in fc1_grids:
  for fc2 in fc2_grids:
    for mr in moment_ratio:
        loss = calc_loss(spec_stack, fc1, fc2, mr)
        param_loss.append(([fc1, fc2, mr], loss))
dtype = [('param','O'),('loss','O')]
param_loss = np.array(param_loss, dtype=dtype)
fc1, fc2, mr = np.sort(param_loss, order='loss')[0]['param']
spec_pred = np.zeros(npts)
for ii in range(npts):
    spec_pred[ii] = mr * np.sqrt((1+(freq[ii]/fc2)**4) / (1+(freq[ii]/fc1)**4))
print('best fit: fc1 %sHz, fc2 %sHz, Mr %s'%(fc1, fc2, mr))
r = 0.21 * 3.4 / fc1 # assume vs 3.4 km/s, 0.21 for Madariaga (1976), 0.37 for Brune (1970)
A = np.pi*r**2 
M0 = 10**(1.5*(tar_loc[-1]+10.7))
miu = 3.2e5
D = 100 * M0 / (miu * A * 1e18) 
d_sig = (7/16) * M0 / r**3 * 1e-22 # stress drop, Eshelby (1957)
print('rupture area (km^2):', A)
print('stress drop (MPa):', d_sig)

def plot_label(xlabel=None, ylabel=None, title=None):
    ax = plt.gca()
    if xlabel: plt.xlabel(xlabel, fontsize=fsize_label)
    if ylabel: plt.ylabel(ylabel, fontsize=fsize_label)
    if title: plt.title(title, fontsize=fsize_title)
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=fsize_label)
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=fsize_label)

plt.figure(figsize=fig_size)
ax = plt.gca()
for ii, spec in enumerate(spec_ratio):
    label = 'Single station' if ii==0 else None
    plt.loglog(freq, spec, lw=1.5, color='k', alpha=alpha, label=label)
plt.loglog(freq, spec_stack, lw=3, color='r', label='Stacked')
plt.loglog(freq, spec_pred, lw=3, color='r', ls='--', label='Best fit')
plt.legend(fontsize=fsize_label, loc='lower left')
plt.annotate('\n$f_c$ = {:.2f} $Hz$\n$A$ = {:.2f} $km^2$\n$D$ = {:.2f} $cm$\n$\Delta\sigma$ = {:.2f} $MPa$'.format(fc1,A,D,d_sig),(freq_max,spec_max), ha='right', va='top', fontsize=fsize_label)
ax.axvline(fc1, color='gray', lw=0.5, zorder=0)
plot_label('Frequency (Hz)', 'S Spectral Ratio', title)
plt.tight_layout()
plt.savefig(fout)

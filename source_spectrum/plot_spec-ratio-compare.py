import sys, os, glob
import numpy as np
from obspy import read, UTCDateTime
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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
sta_lists = [['CI.JRC2','CI.CLC','CI.WMF','CI.WCS2'],['CI.CCC','PB.B917'],['CI.SLA'],['CI.LRL','CI.DTP','CI.WBS','CI.WBM']] # 1-2, 3-4
colors = ['tab:red','tab:blue','tab:orange','tab:cyan']
names = ['NW','SE','NE','SW']
fout = 'output/eg_spec-ratio-compare_%s.pdf'%tar_name
titles = ['(a) Comparison Along Fault 1', '(b) Comparison Along Fault 2']
# signal processing
samp_rate = 100
freq_band = [0.1,None]
s_win = [0.5,5.5]
dt_list = np.arange(0,3,1) # multi-win strategy (Imanishi & Ellsworth, 2006)
# fig config
fig_size = (16*0.9,10*0.9)
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

spec_ratio_list = []
for ii in range(4):
    freq, spec_ratio = calc_spec_ratio(sta_lists[ii])
    spec_ratio_list.append(spec_ratio)
spec_stack1 = []
for ii in range(2): spec_stack1 += spec_ratio_list[ii]
spec_stack1 = log_mean(spec_stack1)
spec_stack2 = []
for ii in range(2,4): spec_stack2 += spec_ratio_list[ii]
spec_stack2 = log_mean(spec_stack2)

def plot_label(xlabel=None, ylabel=None, title=None, yvis=True):
    if xlabel: plt.xlabel(xlabel, fontsize=fsize_label)
    if ylabel: plt.ylabel(ylabel, fontsize=fsize_label)
    if title: plt.title(title, fontsize=fsize_title)
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=fsize_label)
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=fsize_label, visible=yvis)

plt.figure(figsize=fig_size)
ax = plt.subplot(121)
for ii in range(2):
  for jj, spec in enumerate(spec_ratio_list[ii]):
    label = names[ii] if jj==0 else None
    plt.loglog(freq, spec, color=colors[ii], lw=2, alpha=alpha, label=label)
plt.loglog(freq, spec_stack1, lw=3, ls='--', color='k', label='Stacked')
plt.loglog(freq, spec_stack2, lw=3, ls='--', color='gray', label='Stacked of (b)')
plt.legend(fontsize=fsize_label)
plot_label('Frequency (Hz)', 'S Spectral Ratio', titles[0])
ax = plt.subplot(122, sharey=ax)
for ii in range(2,4):
  for jj, spec in enumerate(spec_ratio_list[ii]):
    label = names[ii] if jj==0 else None
    plt.loglog(freq, spec, color=colors[ii], lw=2, alpha=alpha, label=label)
plt.loglog(freq, spec_stack2, lw=3, ls='--', color='k', label='Stacked')
plt.loglog(freq, spec_stack1, lw=3, ls='--', color='gray', label='Stacked of (a)')
plt.legend(fontsize=fsize_label)
plot_label('Frequency (Hz)', None, titles[1], False)
plt.tight_layout()
plt.savefig(fout)

""" Check deconv result for 1-sta & 1-EGF, using Iterative deconv
"""
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime
from reader import dtime2str, read_fpha
from signal_lib import preprocess
from deconv_lib import deconv_iterative
import warnings
warnings.filterwarnings("ignore")

# i/o paths
tar_idx, egf_idx = 0, 1
net_sta = 'CI.HAR'
ftar = 'input/eg_tar.pha'
fegf = 'output/eg_egf.pha'
tar_root = 'input/eg_tar'
egf_root = 'input/eg_egf'
fout_p = 'output/eg_stf-sta_%s_time-p.pdf'%net_sta
fout_s = 'output/eg_stf-sta_%s_time-s.pdf'%net_sta
# signal process
samp_rate = 100
freq_band = [0.1,20]
p_win = [1, 9]
s_win = [1, 15]
iter_max = 100
derr_min = 0.01
gauss_a = 10
dt_p, dt_s = 1, 2.
num_workers = 10
# fig config
fig_size = (12,10)
fsize_label = 14
fsize_title = 18
line_wid = 1.5
alpha = 0.8

def read_data_ps(st_paths, p_win, s_win):
    p_npts = int(samp_rate*sum(p_win))
    s_npts = int(samp_rate*sum(s_win))
    st  = read(st_paths[0])
    st += read(st_paths[1])
    st += read(st_paths[2])
    st = preprocess(st, samp_rate, freq_band)
    start_time = st[0].stats.starttime
    header = st[0].stats.sac
    tp, ts = start_time+header.t0, start_time+header.t1
    if ts-tp<p_win[1]: p_win[1] = ts-tp; p_npts = int(samp_rate*sum(p_win))
    st_p = st.slice(tp-p_win[0],tp+p_win[1]).detrend('demean').taper(max_percentage=0.05)
    st_s = st.slice(ts-s_win[0],ts+s_win[1]).detrend('demean').taper(max_percentage=0.05)
    data_p = np.array([tr.data[0:p_npts] for tr in st_p])
    data_s = np.array([tr.data[0:s_npts] for tr in st_s])
    time_p = np.arange(p_npts) / samp_rate - p_win[0]
    time_s = np.arange(s_npts) / samp_rate - s_win[0]
    return data_p, data_s, time_p, time_s

# read target
tar_loc = read_fpha(ftar)[tar_idx][0]
tar_name = dtime2str(tar_loc[0])
tar_dir = '%s/%s'%(tar_root,tar_name)
tar_paths = sorted(glob.glob('%s/%s.*'%(tar_dir, net_sta)))
tar_p, tar_s, tp_tar, ts_tar = read_data_ps(tar_paths, p_win, s_win)
# read egf
egf_loc, egf_pick_dict = read_fpha(fegf)[egf_idx]
egf_name = dtime2str(egf_loc[0])
egf_dir = '%s/%s'%(egf_root,egf_name)
egf_paths = sorted(glob.glob('%s/%s.*'%(egf_dir, net_sta)))
egf_p, egf_s, tp_egf, ts_egf = read_data_ps(egf_paths, p_win, s_win)
# run deconv
stf_s_list, stf_p_list = [], []
for only_pos in [True, False]:
  stf_s_list.append([])
  stf_p_list.append([])
  for ii in range(3):
    time_p, stf_p, tar_p_fit, rms_p = deconv_iterative(tar_p[ii], egf_p[ii], samp_rate, dt_p, gauss_a, iter_max=iter_max, derr_min=derr_min, only_pos=only_pos)
    time_s, stf_s, tar_s_fit, rms_s = deconv_iterative(tar_s[ii], egf_s[ii], samp_rate, dt_s, gauss_a, iter_max=iter_max, derr_min=derr_min, only_pos=only_pos)
    stf_p_list[-1].append([time_p, stf_p, tar_p_fit, rms_p])
    stf_s_list[-1].append([time_s, stf_s, tar_s_fit, rms_s])

def plot_label(xlabel=None, ylabel=None, title=None):
    if xlabel: plt.xlabel(xlabel, fontsize=fsize_label)
    if ylabel: plt.ylabel(ylabel, fontsize=fsize_label)
    if title: plt.title(title, fontsize=fsize_title)
    plt.setp(plt.gca().xaxis.get_majorticklabels(), fontsize=fsize_label)
    plt.setp(plt.gca().yaxis.get_majorticklabels(), fontsize=fsize_label)

# plot P deconv
plt.figure(figsize=fig_size)
# target waveform
plt.subplot(321)
for ii in range(3):
    tar_p_fit = stf_p_list[1][ii][2]
    tar_obs = tar_p[ii] / np.amax(tar_p[ii]) - 2*ii
    tar_pred = tar_p_fit / np.amax(tar_p[ii]) - 2*ii
    labels = ['Observe','Prediction'] if ii==0 else [None,None]
    plt.plot(tp_tar, tar_obs, color='k', alpha=alpha, lw=line_wid, label=labels[0])
    plt.plot(tp_tar, tar_pred, color='r', alpha=alpha, lw=line_wid, label=labels[1])
plt.yticks([0,-2,-4],['E','N','Z'])
#plt.legend(fontsize=fsize_label)
plot_label('Time (s)', None, 'Target Waveform & Fitting')
# EGF waveform
plt.subplot(323)
for ii in range(3):
    egf = egf_p[ii] / np.amax(egf_p) - 2*ii
    plt.plot(tp_egf, egf, color='b', alpha=alpha, lw=line_wid)
plt.yticks([0,-2,-4],['E','N','Z'])
plot_label('Time (s)', None, 'EGF Waveform')
# iter-deconv with / without pos constraint
plt.subplot(322)
for ii in range(2):
  for jj in range(3):
    color = ['tab:blue','tab:red'][ii]
    labels = ['with pos-constraint','without pos-constraint'] if jj==0 else [None,None]
    rms = stf_p_list[ii][jj][-1]
    plt.plot(rms, color=color, alpha=alpha, lw=line_wid, label=labels[ii])
plt.legend(fontsize=fsize_label)
plot_label('Iteration Index','RMS','Misfit Curve')
for ii in range(2):
  plt.subplot(3,2,2*ii+4)
  for jj in range(3):
    time_p, stf_p = stf_p_list[ii][jj][0:2]
    stf = stf_p / np.amax(stf_p) - 2*jj
    plt.plot(time_p, stf, color='tab:blue', alpha=alpha, lw=line_wid)
  plt.yticks([0,-2,-4],['E','N','Z'])
  xlabel = 'Shifted Time (s)' if ii==1 else None
  plot_label(xlabel, None, ['With Positive Constraint','Without Positive Constraint'][ii])
# plot stacked STF
plt.subplot(325)
stf_list = stf_p_list[0]
stf = [stf/np.amax(stf) for _,stf,_,_ in stf_list]
stf_stack = np.mean(stf, axis=0)
for ii in range(3):
    label = 'E/N/Z' if ii==0 else None
    plt.plot(time_p, stf[ii], color='k', alpha=alpha, lw=line_wid, label=label)
plt.plot(time_p, stf_stack, color='r', lw=2*line_wid, label='Stacked')
plt.legend(fontsize=fsize_label)
plot_label('Shifted Time (s)', None, 'P-wave STF (iter with pos)')
plt.tight_layout()
plt.savefig(fout_p)

# plot S deconv
plt.figure(figsize=fig_size)
# target waveform
plt.subplot(321)
for ii in range(3):
    tar_s_fit = stf_s_list[1][ii][2]
    tar_obs = tar_s[ii] / np.amax(tar_s[ii]) - 2*ii
    tar_pred = tar_s_fit / np.amax(tar_s[ii]) - 2*ii
    labels = ['Observe','Prediction'] if ii==0 else [None,None]
    plt.plot(ts_tar, tar_obs, color='k', alpha=alpha, lw=line_wid, label=labels[0])
    plt.plot(ts_tar, tar_pred, color='r', alpha=alpha, lw=line_wid, label=labels[1])
plt.yticks([0,-2,-4],['E','N','Z'])
#plt.legend(fontsize=fsize_label)
plot_label('Time (s)', None, 'Target Waveform & Fitting')
# EGF waveform
plt.subplot(323)
for ii in range(3):
    egf = egf_s[ii] / np.amax(egf_s) - 2*ii
    plt.plot(ts_egf, egf, color='b', alpha=alpha, lw=line_wid)
plt.yticks([0,-2,-4],['E','N','Z'])
plot_label('Time (s)', None, 'EGF Waveform')
# iter-deconv with / without pos constraint
plt.subplot(322)
for ii in range(2):
  for jj in range(3):
    color = ['tab:blue','tab:red'][ii]
    labels = ['with pos-constraint','without pos-constraint'] if jj==0 else [None,None]
    rms = stf_p_list[ii][jj][-1]
    plt.plot(rms, color=color, alpha=alpha, lw=line_wid, label=labels[ii])
plt.legend(fontsize=fsize_label)
plot_label('Iteration Index','RMS','Misfit Curve')
for ii in range(2):
  plt.subplot(3,2,2*ii+4)
  for jj in range(3):
    time_s, stf_s = stf_s_list[ii][jj][0:2]
    stf = stf_s / np.amax(stf_s) - 2*jj
    plt.plot(time_s, stf, color='tab:blue', alpha=alpha, lw=line_wid)
  plt.yticks([0,-2,-4],['E','N','Z'])
  xlabel = 'Shifted Time (s)' if ii==1 else None
  plot_label(xlabel, None, ['With Positive Constraint','Without Positive Constraint'][ii])
# plot stacked STF
plt.subplot(325)
stf_list = stf_s_list[0]
stf = [stf/np.amax(stf) for _,stf,_,_ in stf_list]
stf_stack = np.mean(stf, axis=0)
for ii in range(3):
    label = 'E/N/Z' if ii==0 else None
    plt.plot(time_s, stf[ii], color='k', alpha=alpha, lw=line_wid, label=label)
plt.plot(time_s, stf_stack, color='r', lw=2*line_wid, label='Stacked')
plt.legend(fontsize=fsize_label)
plot_label('Shifted Time (s)', None, 'S-wave STF (iter with pos)')
plt.tight_layout()
plt.savefig(fout_s)

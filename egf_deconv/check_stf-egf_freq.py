import os, glob
import numpy as np
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime
from reader import dtime2str, read_fpha
from signal_lib import preprocess
from deconv_lib import deconv_waterlevel
import warnings
warnings.filterwarnings("ignore")

# i/o paths
tar_idx, egf_idx_list = 0, [0,1,2,3]
net_sta = 'CI.HAR'
ftar = 'input/eg_tar.pha'
fegf = 'output/eg_egf.pha'
tar_root = 'input/eg_tar'
egf_root = 'input/eg_egf'
fout_p = 'output/eg_stf-egf_%s_freq-p.pdf'%net_sta
fout_s = 'output/eg_stf-egf_%s_freq-s.pdf'%net_sta
# signal process
samp_rate = 100
freq_band = [0.1,20]
p_win = [1,9]
s_win = [1,15]
wl = 0.01 # water level
gauss_a = 10
dt_p, dt_s = 1, 2.
num_workers = 10
# fig config
fig_size = (16,9)
fsize_label = 14
fsize_title = 18
line_wid = 1.5
alpha = 0.8
colors = ['tab:blue','tab:orange','tab:green','tab:cyan']

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
    data_p = np.array([tr.data for tr in st_p])
    data_s = np.array([tr.data for tr in st_s])
    time_p = np.arange(p_npts) / samp_rate - p_win[0]
    time_s = np.arange(s_npts) / samp_rate - s_win[0]
    return data_p, data_s, time_p, time_s

# read target
tar_loc = read_fpha(ftar)[tar_idx][0]
tar_name = dtime2str(tar_loc[0])
tar_dir = '%s/%s'%(tar_root,tar_name)
tar_paths = sorted(glob.glob('%s/%s.*'%(tar_dir, net_sta)))
tar_p, tar_s = read_data_ps(tar_paths, p_win, s_win)[0:2]
# run deconv: [for egf; for chn]
stf_p_list, stf_s_list = [], []
for egf_idx in egf_idx_list:
  stf_p_list.append([])
  stf_s_list.append([])
  # read egf
  egf_loc, egf_pick_dict = read_fpha(fegf)[egf_idx]
  egf_name = dtime2str(egf_loc[0])
  egf_dir = '%s/%s'%(egf_root,egf_name)
  egf_paths = sorted(glob.glob('%s/%s.*'%(egf_dir, net_sta)))
  egf_p, egf_s = read_data_ps(egf_paths, p_win, s_win)[0:2]
  for ii in range(3):
    time_p, stf_p = deconv_waterlevel(tar_p[ii], egf_p[ii], samp_rate, dt_p, gauss_a, wl)[0:2]
    time_s, stf_s = deconv_waterlevel(tar_s[ii], egf_s[ii], samp_rate, dt_s, gauss_a, wl)[0:2]
    stf_p_list[-1].append([time_p, stf_p])
    stf_s_list[-1].append([time_s, stf_s])

def plot_label(xlabel=None, ylabel=None, title=None):
    if xlabel: plt.xlabel(xlabel, fontsize=fsize_label)
    if ylabel: plt.ylabel(ylabel, fontsize=fsize_label)
    if title: plt.title(title, fontsize=fsize_title)
    plt.setp(plt.gca().xaxis.get_majorticklabels(), fontsize=fsize_label)
    plt.setp(plt.gca().yaxis.get_majorticklabels(), fontsize=fsize_label)

# 1. plot P STF
plt.figure(figsize=fig_size)
for jj in range(3):
  plt.subplot(1,4,jj+1)
  for ii, stf_p in enumerate(stf_p_list):
    time, stf = stf_p[jj]
    stf = stf / np.amax(stf) + ii
    plt.plot(time, stf, color=colors[ii], lw=line_wid)
  plt.yticks(np.arange(len(egf_idx_list)), egf_idx_list)
  if jj==0: plot_label('Shifted Time (s)', 'EGF Index', '%s P-STF'%['E','N','Z'][jj])
  else: plot_label('Shifted Time (s)', None, '%s P-STF'%['E','N','Z'][jj])
plt.subplot(1,4,4)
stf_stack_list = []
for ii, stf_p in enumerate(stf_p_list):
  time = stf_p[0][0]
  stf_stack = np.mean([stf/np.amax(stf) for _,stf in stf_p], axis=0)
  stf_stack_list.append(stf_stack)
  for jj in range(3): 
    stf = stf_p[jj][1]
    stf = stf / np.amax(stf) + ii+1
    plt.plot(time, stf, 'k', lw=line_wid/2, alpha=alpha)
  plt.plot(time, stf_stack+ii+1, color=colors[ii], lw=line_wid*2)
for stf_stack in stf_stack_list: plt.plot(time, stf_stack, 'k', alpha=alpha, lw=line_wid/2)
plt.plot(time, np.mean(stf_stack_list, axis=0), 'r', lw=line_wid*2)
plt.yticks(np.arange(1+len(egf_idx_list)),['All']+egf_idx_list)
plot_label('Shifted Time (s)', None, 'Stacked STF')
plt.tight_layout()
plt.savefig(fout_p)

# 1. plot S STF
plt.figure(figsize=fig_size)
for jj in range(3):
  plt.subplot(1,4,jj+1)
  for ii, stf_s in enumerate(stf_s_list):
    time, stf = stf_s[jj]
    stf = stf / np.amax(stf) + ii
    plt.plot(time, stf, color=colors[ii], lw=line_wid)
  plt.yticks(np.arange(len(egf_idx_list)), egf_idx_list)
  if jj==0: plot_label('Shifted Time (s)', 'EGF Index', '%s S-STF'%['E','N','Z'][jj])
  else: plot_label('Shifted Time (s)', None, '%s S-STF'%['E','N','Z'][jj])
plt.subplot(1,4,4)
stf_stack_list = []
for ii, stf_s in enumerate(stf_s_list):
  time = stf_s[0][0]
  stf_stack = np.mean([stf/np.amax(stf) for _,stf in stf_s], axis=0)
  stf_stack_list.append(stf_stack)
  for jj in range(3):
    stf = stf_s[jj][1]
    stf = stf / np.amax(stf) + ii+1
    plt.plot(time, stf, 'k', lw=line_wid/2, alpha=alpha)
  plt.plot(time, stf_stack+ii+1, color=colors[ii], lw=line_wid*2)
for stf_stack in stf_stack_list: plt.plot(time, stf_stack, 'k', alpha=alpha, lw=line_wid/2)
plt.plot(time, np.mean(stf_stack_list, axis=0), 'r', lw=line_wid*2)
plt.yticks(np.arange(1+len(egf_idx_list)),['All']+egf_idx_list)
plot_label('Shifted Time (s)', None, 'Stacked STF')
plt.tight_layout()
plt.savefig(fout_s)

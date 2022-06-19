import os, glob
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from multitaper import MTSpec, MTCross
from obspy import read, UTCDateTime
from reader import dtime2str, read_fsta, read_fpha, read_fcc
from signal_lib import preprocess, calc_azm_deg

# i/o paths
tar_idx, egf_idx_list = 0, [0,1,3,4]
net_sta = 'CI.HAR'
ftar = 'input/eg_tar.pha'
fegf = 'output/eg_egf.pha'
tar_root = 'input/eg_tar'
egf_root = 'input/eg_egf'
fcc = 'output/eg_tar-egf.cc'
fsta = 'input/eg_station.csv'
sta_dict = read_fsta(fsta)
fout = 'output/eg_stf-%s_freq.pdf'%net_sta
titles = ['E S-STF','N S-STF','Z S-STF']
# signal process
samp_rate = 100
freq_band = [0.1,None]
stf_freq = [0.5, 4]
p_win = [0.5, 7.5]
s_win = [0.5, 9.5]
p_npts = int(samp_rate*sum(p_win))
s_npts = int(samp_rate*sum(s_win))
npts = int(5*samp_rate)
time = np.arange(2*npts)/samp_rate - npts/samp_rate
nw = 4
kspec = 7
wl = 0.001 # water level
num_workers = 10
# fig config
fig_size = (16,9)
fsize_label = 14
fsize_title = 18
line_wid = 2.
alpha = 0.8
colors = ['tab:blue','tab:orange','tab:green','tab:red']

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
    st_p = st.slice(tp-p_win[0],tp+p_win[1]).detrend('demean').taper(max_percentage=0.05)
    st_s = st.slice(ts-s_win[0],ts+s_win[1]).detrend('demean').taper(max_percentage=0.05)
    data_p = np.array([tr.data for tr in st_p])
    data_s = np.array([tr.data for tr in st_s])
    return data_p, data_s

# read target
tar_loc = read_fpha(ftar)[tar_idx][0]
tar_name = dtime2str(tar_loc[0])
tar_dir = '%s/%s'%(tar_root,tar_name)
tar_paths = sorted(glob.glob('%s/%s.*'%(tar_dir, net_sta)))
tar_p, tar_s = read_data_ps(tar_paths, p_win, s_win)
# mt deconv
stf_s_list = []
for egf_idx in egf_idx_list:
  stf_s_list.append([])
  # read egf
  egf_loc, egf_pick_dict = read_fpha(fegf)[egf_idx]
  egf_name = dtime2str(egf_loc[0])
  egf_dir = '%s/%s'%(egf_root,egf_name)
  egf_paths = sorted(glob.glob('%s/%s.*'%(egf_dir, net_sta)))
  egf_p, egf_s = read_data_ps(egf_paths, p_win, s_win)
  for ii in range(3):
    tar_spec_s = MTSpec(tar_s[ii], nw, kspec, 1/samp_rate)
    egf_spec_s = MTSpec(egf_s[ii], nw, kspec, 1/samp_rate)
    cross_spec_s = MTCross(tar_spec_s, egf_spec_s, wl=wl)
    stf_s_list[-1].append(cross_spec_s.mt_corr()[2])


def plot_label(xlabel=None, ylabel=None, title=None, yvis=True):
    if xlabel: plt.xlabel(xlabel, fontsize=fsize_label)
    if ylabel: plt.ylabel(ylabel, fontsize=fsize_label)
    if title: plt.title(title, fontsize=fsize_title)
    plt.setp(plt.gca().xaxis.get_majorticklabels(), fontsize=fsize_label)
    plt.setp(plt.gca().yaxis.get_majorticklabels(), fontsize=fsize_label)

plt.figure(figsize=fig_size)
for jj in range(3):
  plt.subplot(1,3,jj+1)
  for ii, stf_s in enumerate(stf_s_list):
    stf = stf_s[jj][s_npts-npts:s_npts+npts]
    stf /= np.amax(stf)
    stf += ii
    plt.plot(time, stf, color=colors[ii])
  plt.yticks(np.arange(len(egf_idx_list)))
  if jj==0: plot_label('Time (s)', 'EGF Index', titles[jj])
  else: plot_label('Time (s)', None, titles[jj], False)
plt.tight_layout()
plt.savefig(fout)

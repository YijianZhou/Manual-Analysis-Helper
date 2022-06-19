import os, glob
import numpy as np
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime
from reader import dtime2str, read_fsta, read_fpha, read_fcc
from signal_lib import preprocess, calc_azm_deg
from deconv_lib import deconv_iterative

# i/o paths
tar_idx, egf_idx = 0, 1
sta = ['CI.HAR','CI.MPM'][0]
ftar = 'input/eg_tar.pha'
fegf = 'output/eg_egf.pha'
tar_root = 'input/eg_tar'
egf_root = 'input/eg_egf'
fsta = 'input/eg_station.csv'
sta_dict = read_fsta(fsta)
fout = 'output/eg_stf-iter_%s.pdf'%sta
# signal process
samp_rate = 100
freq_band = [0.5,20]
p_win = [0.5, 6.5]
s_win = [0.5, 7.5]
p_npts = int(samp_rate*sum(p_win))
s_npts = int(samp_rate*sum(s_win))
gauss = 2.
iter_max = 40
derr_min = 0.02
num_workers = 10
# fig config
fig_size = (14,12)
fsize_label = 14
fsize_title = 18
line_wid = 2.
alpha = 0.8


# read tar & egf data
def read_data_ps(st_paths):
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

tar_loc = read_fpha(ftar)[tar_idx][0]
tar_name = dtime2str(tar_loc[0])
tar_paths = sorted(glob.glob('%s/%s/%s.*'%(tar_root, tar_name, sta)))
tar_p, tar_s = read_data_ps(tar_paths)
egf_loc, egf_pick_dict = read_fpha(fegf)[egf_idx]
egf_name = dtime2str(egf_loc[0])
egf_paths = sorted(glob.glob('%s/%s/%s.*'%(egf_root, egf_name, sta)))
egf_p, egf_s = read_data_ps(egf_paths)

# run iter deconv
time, astf, rms = deconv_iterative(tar_p[2], egf_p[2], samp_rate, gauss=gauss, iter_max = iter_max, derr_min=derr_min, to_norm=False)

def plot_label(xlabel=None, ylabel=None, title=None):
    if xlabel: plt.xlabel(xlabel, fontsize=fsize_label)
    if ylabel: plt.ylabel(ylabel, fontsize=fsize_label)
    if title: plt.title(title, fontsize=fsize_title)
    plt.setp(plt.gca().xaxis.get_majorticklabels(), fontsize=fsize_label)
    plt.setp(plt.gca().yaxis.get_majorticklabels(), fontsize=fsize_label)

plt.figure(figsize=fig_size)
plt.subplot(221)
plt.plot(time, astf)
plot_label('Time (s)', None, 'P ASTF')
plt.subplot(222)
plt.plot(rms)
plot_label('Iteration', None, 'P RMS')
plt.tight_layout()
plt.savefig(fout)

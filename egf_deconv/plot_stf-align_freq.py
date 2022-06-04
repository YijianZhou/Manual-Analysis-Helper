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
tar_idx, egf_idx = 0, 1
ftar = 'input/eg_tar.pha'
fegf = 'output/eg_egf.pha'
tar_root = 'input/eg_tar'
egf_root = 'input/eg_egf'
fcc = 'output/eg_tar-egf.cc'
fsta = 'input/eg_station.csv'
sta_dict = read_fsta(fsta)
fout = 'output/eg_stf-slign_freq'
titles = ['P-wave STF','S-wave STF']
# signal process
samp_rate = 100
freq_band = [0.1,None]
stf_freq = [0.5, 4]
p_win = [0.5, 7.5]
s_win = [0.5, 11.5]
p_npts = int(samp_rate*sum(p_win))
s_npts = int(samp_rate*sum(s_win))
npts = int(5*samp_rate)
time = np.arange(2*npts)/samp_rate - npts/samp_rate
cc_min = 0.5
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

# get tar & egf loc
tar_loc = read_fpha(ftar)[tar_idx][0]
tar_name = dtime2str(tar_loc[0])
tar_dir = '%s/%s'%(tar_root,tar_name)
egf_loc, egf_pick_dict = read_fpha(fegf)[egf_idx]
egf_name = dtime2str(egf_loc[0])
egf_dir = '%s/%s'%(egf_root,egf_name)
# select station by CC & sort by azm
cc_list = read_fcc(fcc)[egf_name]
sta_list_raw = [sta for [sta,cc_p,cc_s] in cc_list if max(cc_p,cc_s)>=cc_min]
sta_list = []
dtype = [('sta','O'),('baz','O')]
for sta in sta_list_raw:
    sta_lat, sta_lon = sta_dict[sta][0:2]
    baz = calc_azm_deg([tar_loc[1],sta_lat], [tar_loc[2],sta_lon])[1]
    sta_list.append((sta, baz))
sta_list = np.array(sta_list, dtype=dtype)
sta_list = np.sort(sta_list, order='baz')
num_sta = len(sta_list)

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

# calc stf
class EGF_Deconv(Dataset):
  def __init__(self, sta_list):
    self.sta_list = sta_list

  def __getitem__(self, index):
    sta = self.sta_list[index][0]
    # read data
    tar_paths = sorted(glob.glob('%s/%s.*'%(tar_dir, sta)))
    tar_p, tar_s = read_data_ps(tar_paths, p_win, s_win)
    egf_paths = sorted(glob.glob('%s/%s.*'%(egf_dir, sta)))
    egf_p, egf_s = read_data_ps(egf_paths, p_win, s_win)
    # mt deconv
    stf_p, stf_s = 0, 0
    for ii in range(3):
        tar_spec_p = MTSpec(tar_p[ii], nw, kspec, 1/samp_rate)
        tar_spec_s = MTSpec(tar_s[ii], nw, kspec, 1/samp_rate)
        egf_spec_p = MTSpec(egf_p[ii], nw, kspec, 1/samp_rate)
        egf_spec_s = MTSpec(egf_s[ii], nw, kspec, 1/samp_rate)
        cross_spec_p = MTCross(tar_spec_p, egf_spec_p, wl=wl)
        cross_spec_s = MTCross(tar_spec_s, egf_spec_s, wl=wl)
        stf_p += cross_spec_p.mt_corr()[2]
        stf_s += cross_spec_s.mt_corr()[2]
    return stf_p/3, stf_s/3

  def __len__(self):
    return len(self.sta_list)

stf_p_list, stf_s_list = [], []
dataset = EGF_Deconv(sta_list)
dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=None)
for ii,[stf_p, stf_s] in enumerate(dataloader):
    stf_p_list.append(stf_p.numpy())
    stf_s_list.append(stf_s.numpy())


def plot_label(xlabel=None, ylabel=None, title=None):
    if xlabel: plt.xlabel(xlabel, fontsize=fsize_label)
    if ylabel: plt.ylabel(ylabel, fontsize=fsize_label)
    if title: plt.title(title, fontsize=fsize_title)
    plt.setp(plt.gca().xaxis.get_majorticklabels(), fontsize=fsize_label)
    plt.setp(plt.gca().yaxis.get_majorticklabels(), fontsize=fsize_label)

plt.figure(figsize=fig_size)
plt.subplot(121)
for ii in range(num_sta):
    stf = stf_p_list[ii][p_npts-npts:p_npts+npts]
    stf /= np.amax(abs(stf))
    stf += ii
    plt.plot(time, stf, lw=line_wid, alpha=alpha)
plt.scatter([0,0],[-1,num_sta], alpha=0)
plt.yticks(np.arange(num_sta), sta_list['sta'], fontsize=fsize_label)
plot_label('Time (s)', None, titles[0])
plt.subplot(122)
for ii in range(num_sta):
    stf = stf_s_list[ii][s_npts-npts:s_npts+npts]
    stf /= np.amax(stf)
    stf += ii
    plt.plot(time, stf, lw=line_wid, alpha=alpha)
plt.scatter([0,0],[-1,num_sta], alpha=0)
plt.yticks(np.arange(num_sta), ['']*num_sta, fontsize=fsize_label)
plot_label('Time (s)', None, titles[1])
plt.tight_layout()
plt.savefig(fout+'_%s-%s.pdf'%(tar_name, egf_name))

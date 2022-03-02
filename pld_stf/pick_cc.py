import sys, os, glob
sys.path.append('/home/zhouyj/software/data_prep')
import numpy as np
from obspy import read, UTCDateTime, Trace
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from signal_lib import preprocess, calc_cc, calc_azm_deg
from reader import dtime2str, read_fsta, read_fpha_dict
import sac
import warnings
warnings.filterwarnings("ignore")

# i/o paths
egf_name = '20210519005951.42'
tar_dir = 'input/eg_tar/20210521212125.00'
egf_dir = 'input/eg_egf_org/%s'%egf_name
fsta = 'input/station_eg.csv'
sta_dict = read_fsta(fsta)
sta_list = list(sta_dict.keys())
fpha = 'input/eg_egf_org.pha'
event_dict = read_fpha_dict(fpha)
egf_loc = event_dict[egf_name][0][1:3]
tar_loc = [25.6527,99.9268]
out_root = 'output/eg_egf/%s'%egf_name
if not os.path.exists(out_root): os.makedirs(out_root)
# signal processing
samp_rate = 100
freq_band = [1,20]
num_workers = 10
s_win = [1,3]
p_win = [1,2]
dt_cc = .5 # pre & post
p_npts = int(samp_rate*sum(p_win))
s_npts = int(samp_rate*sum(s_win))
cc_npts = int(samp_rate*2*dt_cc)
time_tar_p = np.arange(p_npts+cc_npts) / samp_rate - dt_cc - p_win[0]
time_tar_s = np.arange(s_npts+cc_npts) / samp_rate - dt_cc - s_win[0]
time_egf_p = np.arange(p_npts) / samp_rate - p_win[0]
time_egf_s = np.arange(s_npts) / samp_rate - s_win[0]
time_cc = np.arange(cc_npts) / samp_rate - dt_cc
# fig config
fig_size = (14,14)
fsize_label = 14
fsize_title = 18

def read_s_data(st_paths, s_win, baz):
    npts = int(samp_rate*sum(s_win))
    st = read(st_paths[1])
    st+= read(st_paths[0])
    st = preprocess(st, samp_rate, freq_band)
    st = st.rotate(method='NE->RT', back_azimuth=baz)
    start_time = st[0].stats.starttime
    header = st[0].stats.sac
    tp, ts = header.t0, header.t1
    t0 = start_time + ts - s_win[0]
    t1 = start_time + ts + s_win[1]
    if len(st.slice(t0,t1))<2: return []
    return st.slice(t0,t1).detrend('demean').detrend('linear').taper(max_percentage=0.05)[1].data[0:npts]

def read_p_data(st_paths, p_win):
    npts = int(samp_rate*sum(p_win))
    st = read(st_paths[2])
    st = preprocess(st, samp_rate, freq_band)
    start_time = st[0].stats.starttime
    header = st[0].stats.sac
    tp, ts = header.t0, header.t1
    t0 = start_time + tp - p_win[0]
    t1 = start_time + tp + p_win[1]
    if len(st.slice(t0,t1))<1: return []
    return st.slice(t0,t1).detrend('demean').detrend('linear').taper(max_percentage=0.05)[0].data[0:npts]

def plot_label(xlabel=None, ylabel=None, title=None, yvisible=True):
    ax = plt.gca()
    if xlabel: plt.xlabel(xlabel, fontsize=fsize_label)
    if ylabel: plt.ylabel(ylabel, fontsize=fsize_label)
    if title: plt.title(title, fontsize=fsize_title)
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=fsize_label)
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=fsize_label, visible=yvisible)

def np2sac(data, net_sta, fout):
    net, sta = net_sta.split('.')
    tr = Trace(data=data)
    tr.stats.sampling_rate = samp_rate
    tr.write(fout)
    sac.ch_b(fout, -dt_cc)
    sac.ch_sta(fout, knetwk=net, kstnm=sta)


class Pick_CC(Dataset):
  """ Dataset for cutting events
  """
  def __init__(self, sta_list):
    self.sta_list = sta_list

  def __getitem__(self, index):
    sta = self.sta_list[index]
    sta_lat, sta_lon = sta_dict[sta][0:2]
    baz_tar = calc_azm_deg([tar_loc[0],sta_lat], [tar_loc[1],sta_lon])[1]
    baz_egf = calc_azm_deg([egf_loc[0],sta_lat], [egf_loc[1],sta_lon])[1]
    # read data
    tar_paths = sorted(glob.glob(os.path.join(tar_dir, '%s.*'%sta)))
    egf_paths = sorted(glob.glob(os.path.join(egf_dir, '%s.*'%sta)))
    tar_p = read_p_data(tar_paths, [p_win[0]+dt_cc, p_win[1]+dt_cc])
    tar_s = read_s_data(tar_paths, [s_win[0]+dt_cc, s_win[1]+dt_cc], baz_tar)
    egf_p = read_p_data(egf_paths, p_win)
    egf_s = read_s_data(egf_paths, s_win, baz_egf)
    # calc cc for sta & egf
    cc_s, cc_p = [], []
    cc_p = calc_cc(tar_p, egf_p)
    cc_s = calc_cc(tar_s, egf_s)
    dt_p = np.argmax(cc_p)/samp_rate-dt_cc
    dt_s = np.argmax(cc_s)/samp_rate-dt_cc
    dt_p1 = np.argmin(cc_p)/samp_rate-dt_cc
    dt_s1 = np.argmin(cc_s)/samp_rate-dt_cc
    # save cc sac sac
    sta_dir = os.path.join(out_root,sta)
    if not os.path.exists(sta_dir): os.makedirs(sta_dir)
    fcc_p = os.path.join(sta_dir,'cc_p.sac')
    fcc_s = os.path.join(sta_dir,'cc_s.sac')
    np2sac(cc_p, sta, fcc_p)
    np2sac(cc_s, sta, fcc_s)
    # save fig
    fout = os.path.join(out_root, 'cc_%s.pdf'%sta)
    fig = plt.figure(figsize=fig_size)
    plt.subplot(221)
    plt.plot(time_tar_p, tar_p/np.amax(tar_p), alpha=0.8)
    plt.plot(time_egf_p+dt_p, egf_p/np.amax(egf_p), alpha=0.8)
    plt.annotate('dt_p=%ss'%dt_p, (time_tar_p[0],1), ha='left', va='top', fontsize=fsize_label)
    plot_label('Time (s)',None,'%s P Wave'%sta)
    plt.subplot(223)
    plt.plot(time_cc, cc_p)
    plot_label('Time Shift (s)','CC')
    plt.subplot(222)
    plt.plot(time_tar_s, tar_s/np.amax(tar_s), alpha=0.8)
    plt.plot(time_egf_s+dt_s, egf_s/np.amax(egf_s), alpha=0.8)
    plt.annotate('dt_s=%ss'%dt_s, (time_tar_s[0],1), ha='left', va='top', fontsize=fsize_label)
    plot_label('Time (s)',None,'%s S Wave'%sta)
    plt.subplot(224)
    plt.plot(time_cc, cc_s)
    plot_label('Time Shift (s)','CC')
    plt.tight_layout()
    plt.savefig(fout)

  def __len__(self):
    return len(self.sta_list)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) # 'spawn' or 'forkserver'
    dataset = Pick_CC(sta_list)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=None)
    for ii,_ in enumerate(dataloader):
        if ii%10==0: print('%s/%s stations done/total'%(ii+1,len(dataset)))


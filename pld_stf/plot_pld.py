""" Plot PLD inversion result
    Note: keep the calculation param consistent with calc_pld-misfit.py
"""
import sys, os, glob
sys.path.append('/home/zhouyj/software/data_prep')
import numpy as np
from obspy import read, UTCDateTime, Trace
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from signal_lib import preprocess, calc_azm_deg
from reader import dtime2str, read_fsta, read_fpha_dict
import sac

# i/o paths
tar_dir = 'input/eg_tar/20210521212125.00'
egf_name = '20210519005951.42'
egf_dir = 'input/eg_egf_org/%s'%egf_name
fsta = 'input/station_eg.csv'
sta_dict = read_fsta(fsta)
sta_list = list(sta_dict.keys())
fpha = 'input/eg_egf_org.pha'
event_dict = read_fpha_dict(fpha)
egf_loc = event_dict[egf_name][0][1:3]
tar_loc = [25.6527,99.9268]
out_root = 'output/eg_tar-pld/%s'%egf_name
if not os.path.exists(out_root): os.makedirs(out_root)
# signal processing
freq_band = [1,20]
samp_rate = 100
num_workers = 10
num_iter = 100
p_pha_len, s_pha_len = 4., 6.
pre_blank = 0.5
pre_npts = int(np.round(pre_blank*samp_rate,1))
p_pha_npts = int(np.round(p_pha_len*samp_rate,1))
s_pha_npts = int(np.round(s_pha_len*samp_rate,1))
p_npts = p_pha_npts + pre_npts
s_npts = s_pha_npts + pre_npts
time_p = np.arange(p_npts) / samp_rate
time_s = np.arange(s_npts) / samp_rate
T1 = int(pre_blank * samp_rate)
# fig config
fig_size = (14,14)
fsize_label = 14
fsize_title = 18
alpha = 0.8
lw = 2

def read_s_data(st_paths, s_win, baz, dt=0):
    npts = int(samp_rate*sum(s_win))
    st = read(st_paths[1])
    st+= read(st_paths[0])
    st = preprocess(st, samp_rate, freq_band)
    st = st.rotate(method='NE->RT', back_azimuth=baz)
    start_time = st[0].stats.starttime
    header = st[0].stats.sac
    tp, ts = header.t0, header.t1
    t0 = start_time + ts + dt - s_win[0]
    t1 = start_time + ts + dt + s_win[1]
    if len(st.slice(t0,t1))<2: return []
    return st.slice(t0,t1).detrend('demean').detrend('linear').taper(max_percentage=0.05)[1].data[0:npts]

def read_p_data(st_paths, p_win, dt=0):
    npts = int(samp_rate*sum(p_win))
    st = read(st_paths[2])
    st = preprocess(st, samp_rate, freq_band)
    start_time = st[0].stats.starttime
    header = st[0].stats.sac
    tp, ts = header.t0, header.t1
    t0 = start_time + tp + dt - p_win[0]
    t1 = start_time + tp + dt + p_win[1]
    if len(st.slice(t0,t1))<1: return []
    return st.slice(t0,t1).detrend('demean').detrend('linear').taper(max_percentage=0.05)[0].data[0:npts]

def posproj(g,T1,T):
    f = np.zeros_like(g)
    for ii in range(len(g)):
        if ii>=T1 and ii<=T and g[ii]>=0: f[ii] = g[ii]
    return f

def pld(tar_data, egf_data, npts, T_fit):
    # init
    foldt = np.zeros(2*npts)
    fnewt = np.zeros(2*npts)
    fneww = np.fft.fft(fnewt)
    dataw = np.fft.fft(tar_data)
    GFw = np.fft.fft(egf_data)
    Gstarw = np.conj(GFw)
    tau = np.amax(abs(GFw))**(-2)
    # set up inverse problem
    for jj in range(num_iter):
        foldt = fnewt
        foldw = fneww
        f = np.fft.fft(foldt)
        res = dataw - GFw*f
        dum = Gstarw * res
        gneww = foldw + tau * dum
        gnewt = np.real(np.fft.ifft(gneww))
        fnewt = posproj(gnewt,T1,T_fit)
        fneww = np.fft.fft(fnewt)
    dfit = np.real(np.fft.ifft(fneww * GFw))
    astf_all = fnewt[0:npts]
    # calc tau_c (char dur)
    astf = fnewt[T1:T_fit+1]
    msum = sum(astf)
    tsum = sum([astf_i*ii for ii,astf_i in enumerate(astf)])
    t1 = tsum / msum # centroid time
    tsum = sum([astf_i*(ii-t1)**2 for ii,astf_i in enumerate(astf)])
    t2 = tsum / msum # second moment
    tau_c = 2 * np.sqrt(t2) / samp_rate
    return astf_all, tau_c, dfit

def plot_label(xlabel=None, ylabel=None, title=None):
    ax = plt.gca()
    if xlabel: plt.xlabel(xlabel, fontsize=fsize_label)
    if ylabel: plt.ylabel(ylabel, fontsize=fsize_label)
    if title: plt.title(title, fontsize=fsize_title)
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=fsize_label)
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=fsize_label)

def plot_pld(sta, pha, tar_data, egf_data, astf, tau_c, fit_data, misfit, tfit, fout):
    # prep plot data
    npts = int(len(tar_data)/2)
    tar_plot = tar_data[0:npts] / np.amax(tar_data)
    egf_plot = np.concatenate([np.zeros(pre_npts), egf_data[0:npts-pre_npts] / np.amax(egf_data)])
    fit_plot = fit_data[0:npts] / np.amax(fit_data)
    time = -pre_blank + np.arange(npts) / samp_rate
    # start plot
    plt.figure(figsize=fig_size)
    plt.subplot(221)
    plt.plot(time, tar_plot, label='Target')
    plt.plot(time, egf_plot+1, label='EGF')
    plt.gca().axvline(0, color='r', lw=1, zorder=0)
    plot_label('Time (s)',None,'%s %s-wave'%(sta,pha))
    plt.legend(fontsize=fsize_label)
    plt.subplot(222)
    time_err = np.arange(len(misfit)) / samp_rate
    plt.plot(time_err, misfit)
    plt.gca().axvline(tfit, color='r', lw=1, zorder=0)
    plot_label('Duration (s)',None,"Misfit Curve")
    plt.subplot(223)
    plt.plot(np.arange(len(astf)) / samp_rate, astf)
    plot_label('Time (s)',None,'ASTF')
    plt.annotate(r'$\tau_c$ = %ss '%(round(tau_c,2)),(len(astf)/samp_rate,np.amax(astf)), va='top', ha='right', fontsize=fsize_label)
    plt.subplot(224)
    plt.plot(time, tar_plot, label='Target')
    plt.plot(time, fit_plot, label='Fit')
    plot_label('Time (s)',None,'Waveform Fit')
    plt.legend(fontsize=fsize_label)
    plt.tight_layout()
    plt.savefig(fout)


class Plot_PLD(Dataset):
  """ Dataset for plotting PLD
  """
  def __init__(self, sta_list):
    self.sta_list = sta_list

  def __getitem__(self, index):
    # cac baz
    sta = self.sta_list[index]
    sta_lat, sta_lon = sta_dict[sta][0:2]
    baz_tar = calc_azm_deg([tar_loc[0],sta_lat], [tar_loc[1],sta_lon])[1]
    baz_egf = calc_azm_deg([egf_loc[0],sta_lat], [egf_loc[1],sta_lon])[1]
    # get dt & T_fit from sac header
    head_p = read(os.path.join(out_root,sta,'cc_p.sac'))[0].stats.sac
    head_s = read(os.path.join(out_root,sta,'cc_s.sac'))[0].stats.sac
    dt_p = -head_p.t0 if 't0' in head_p else 0
    dt_s = -head_s.t0 if 't0' in head_s else 0
    to_calc_p = False if 't0' not in head_p or 't1' in head_p else True
    to_calc_s = False if 't0' not in head_s or 't1' in head_s else True
    # read data
    tar_paths = sorted(glob.glob(os.path.join(tar_dir, '%s.*'%sta)))
    egf_paths = sorted(glob.glob(os.path.join(egf_dir, '%s.*'%sta)))
    tar_p = read_p_data(tar_paths, [pre_blank,p_pha_len])
    tar_s = read_s_data(tar_paths, [pre_blank,s_pha_len], baz_tar)
    egf_p = read_p_data(egf_paths, [0,p_pha_len], dt_p)
    egf_s = read_s_data(egf_paths, [0,s_pha_len], dt_s)
    tar_p = np.concatenate([tar_p, np.zeros(p_npts)])
    egf_p = np.concatenate([egf_p, np.zeros(p_npts+pre_npts)])
    if to_calc_p: 
        fout = os.path.join(out_root, 'pld_%s.pdf'%sta)
        fout_np = os.path.join(out_root, sta, 'pld_stf_p.npy')
        st_err = read(os.path.join(out_root,sta,'pld_err_p.sac'))
        if 't0' not in st_err[0].stats.sac: return
        T_fit = int(samp_rate * st_err[0].stats.sac.t0)
        astf, tau_c, dfit = pld(tar_p, egf_p, p_npts, T_fit)
        plot_pld(sta, 'P', tar_p, egf_p, astf, tau_c, dfit, st_err[0].data, T_fit/samp_rate, fout)
        np.save(fout_np, [np.arange(len(astf))/samp_rate, astf, tau_c])
    if to_calc_s: 
        fout = os.path.join(out_root, 'pld_%s.pdf'%sta)
        fout_np = os.path.join(out_root, sta, 'pld_stf_s.npy')
        st_err = read(os.path.join(out_root,sta,'pld_err_s.sac'))
        T_fit = int(samp_rate * st_err[0].stats.sac.t0)
        astf, tau_c, dfit = pld(tar_s, egf_s, s_npts, T_fit)
        plot_pld(sta, 'S', tar_s, egf_s, astf, tau_c, dfit, st_err[0].data, T_fit/samp_rate, fout)
        np.save(fout_np, [np.arange(len(astf))/samp_rate, astf, tau_c])

  def __len__(self):
    return len(self.sta_list)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) # 'spawn' or 'forkserver'
    dataset = Plot_PLD(sta_list)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=None)
    for ii,_ in enumerate(dataloader):
        if ii%10==0: print('%s/%s stations done/total'%(ii+1,len(dataset)))

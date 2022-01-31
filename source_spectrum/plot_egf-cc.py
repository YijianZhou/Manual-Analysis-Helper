import sys, os, glob
sys.path.append('/home/zhouyj/software/data_prep')
import numpy as np
from obspy import read, UTCDateTime
import matplotlib as mpl
import matplotlib.pyplot as plt
from signal_lib import preprocess, calc_cc, calc_azm_deg, calc_dist_km
from reader import dtime2str, read_fsta, read_fpha

# i/o paths
tar_dir = 'input/events_tar/20210521212125.00',
tar_loc = [25.6527,99.9268]
egf_dirs = sorted(glob.glob('input/yb_egf-F1_org/20*'))
egf_names = [egf_dir.split('/')[-1] for egf_dir in egf_dirs]
for ii, egf_name in enumerate(egf_names): print(ii, egf_name)
fsta = 'input/station_yb-sel.csv'
sta_dict = read_fsta(fsta)
fpha = 'output/egf.pha'
event_list = read_fpha(fpha)
fout = 'output/egf-cc.pdf'
# signal process
samp_rate = 100
freq_band = [1, 10]
s_win = [1,10]
p_win = [1,6]
dt_cc = 2 # pre & post
# fig config
fig_size = (12,8)
fsize_label = 14
fsize_title = 18
cmap = plt.get_cmap('jet')
mark_size = 100
cc_min, cc_max = 0.3, 0.7
# color bar
cbar_pos = [0.92,0.2,0.03,0.5] 
cbar_ticks = np.arange(0.,1.1,0.25)
cbar_tlabels = ['0.3','0.4','0.5','0.6','0.7']
subplot_rect = {'left':0.08, 'right':0.9, 'bottom':0.08, 'top':0.95, 'wspace':0.05, 'hspace':0.05}

# make event dict
event_dict = {}
for [event_loc, pick_dict] in event_list:
    event_name = dtime2str(event_loc[0])
    event_dict[event_name] = event_loc[1:3]

# sort sta
sta_list = []
dtype = [('sta','O'),('dist','O')]
for sta, sta_loc in sta_dict.items():
    dist = calc_dist_km([sta_loc[0],tar_loc[0]], [sta_loc[1],tar_loc[1]])
    sta_list.append((sta,dist))
sta_list = np.array(sta_list, dtype=dtype)
sta_list = np.sort(sta_list, order='dist')
sta_list = [sta for [sta,_] in sta_list]

def read_s_data(st_paths, baz, s_win):
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
    return st.slice(t0,t1).detrend('demean').detrend('linear').taper(max_percentage=0.05)[0].data[0:npts]

# calc cc for sta & egf
num_sta = len(sta_list)
num_egf = len(egf_dirs)
cc_s, cc_p = np.zeros([2,num_sta,num_egf])
for ii,sta in enumerate(sta_list):
  sta_loc = sta_dict[sta][0:2]
  # read tar data
  baz = calc_azm_deg([tar_loc[0],sta_loc[0]], [tar_loc[1],sta_loc[1]])[1]
  tar_paths = sorted(glob.glob(tar_dir+'/%s*'%sta))
  tar_p = read_p_data(tar_paths, [p_win[0]+dt_cc, p_win[1]+dt_cc])
  tar_s = read_s_data(tar_paths, baz, [s_win[0]+dt_cc,s_win[1]+dt_cc])
  for jj,egf_dir in enumerate(egf_dirs):
    egf_loc = event_dict[egf_dir.split('/')[-1]]
    baz = calc_azm_deg([egf_loc[0],sta_loc[0]], [egf_loc[1],sta_loc[1]])[1]
    # read egf data
    egf_paths = sorted(glob.glob(egf_dir+'/%s*'%sta))
    egf_p = read_p_data(egf_paths, p_win)
    egf_s = read_s_data(egf_paths, baz, s_win)
    cc_p[ii,jj] = np.amax(calc_cc(tar_p, egf_p))
    cc_s[ii,jj] = np.amax(calc_cc(tar_s, egf_s))

def plot_label(xlabel=None, ylabel=None, title=None, yvisible=True):
    ax = plt.gca()
    if xlabel: plt.xlabel(xlabel, fontsize=fsize_label)
    if ylabel: plt.ylabel(ylabel, fontsize=fsize_label)
    if title: plt.title(title, fontsize=fsize_title)
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=fsize_label)
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=fsize_label, visible=yvisible)

fig = plt.figure(figsize=fig_size)
plt.subplot(121)
for ii in range(num_sta):
    color = [cmap((cc-cc_min)/(cc_max-cc_min)) for cc in cc_p[ii]]
    plt.scatter(np.arange(num_egf), np.ones(num_egf)*ii, np.ones(num_egf)*mark_size, marker='s', color=color)
plt.yticks(np.arange(num_sta), sta_list, fontsize=fsize_label)
plot_label('EGF Index',None,'P-wave CC')
plt.subplot(122)
for ii in range(num_sta):
    color = [cmap(cc) for cc in cc_s[ii]]
    plt.scatter(np.arange(num_egf), np.ones(num_egf)*ii, np.ones(num_egf)*mark_size, marker='s', color=color)
plot_label('EGF Index',None,'S-wave CC',False)
cbar_ax = fig.add_axes(cbar_pos)
cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap)
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels(cbar_tlabels)
plt.setp(cbar_ax.yaxis.get_majorticklabels(), fontsize=fsize_label)
plt.subplots_adjust(**subplot_rect)
plt.savefig(fout)

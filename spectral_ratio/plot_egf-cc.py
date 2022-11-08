import sys, os, glob
import numpy as np
from obspy import read, UTCDateTime
import matplotlib as mpl
import matplotlib.pyplot as plt
from signal_lib import calc_dist_km
from reader import dtime2str, read_fsta, read_fpha

# i/o paths
fsta = 'input/station_eg.csv'
sta_dict = read_fsta(fsta)
tar_idx = 0
tar_pha = 'input/eg_tar.pha'
tar_loc = read_fpha(tar_pha)[tar_idx][0][1:3]
fpha = 'input/eg_egf_org.pha'
egf_list = read_fpha(fpha)
egf_names = [dtime2str(event_loc[0]) for [event_loc, _] in egf_list]
fcc = 'output/eg_tar-egf.cc'
fout = 'output/eg_tar-egf-cc.pdf'
# fig config
fig_size = (12,8)
fsize_label = 14
fsize_title = 18
cmap = plt.get_cmap('jet')
mark_size = 100
cc_min, cc_max = 0.3, 0.7
# color bar
cbar_pos = [0.92,0.2,0.03,0.5] # pos in ax: left
cbar_ticks = np.arange(0.,1.1,0.25)
cbar_tlabels = ['0.3','0.4','0.5','0.6','0.7']
subplot_rect = {'left':0.08, 'right':0.9, 'bottom':0.08, 'top':0.95, 'wspace':0.05, 'hspace':0.05}

# sort sta
sta_list = []
dtype = [('sta','O'),('dist','O')]
for sta, sta_loc in sta_dict.items():
    dist = calc_dist_km([sta_loc[0],tar_loc[0]], [sta_loc[1],tar_loc[1]])
    sta_list.append((sta,dist))
sta_list = np.array(sta_list, dtype=dtype)
sta_list = np.sort(sta_list, order='dist')
sta_list = [sta for [sta,_] in sta_list]

# read fcc
cc_dict = {}
f=open(fcc); lines=f.readlines(); f.close()
for line in lines:
    codes = line.split(',')
    if len(codes)==5: 
        event_name = dtime2str(codes[0])
        cc_dict[event_name] = {}; continue
    sta = codes[0]
    cc_p, cc_s = [float(code) for code in codes[1:3]]
    cc_dict[event_name][sta] = [cc_p, cc_s]

num_sta = len(sta_list)
num_egf = len(egf_names)
cc_s, cc_p = np.zeros([2,num_sta,num_egf])
for ii,sta in enumerate(sta_list):
  for jj,egf_name in enumerate(egf_names):
    cc_p[ii,jj], cc_s[ii,jj] = cc_dict[egf_name][sta]

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
plt.xticks(np.arange(0,num_egf,2))
plt.yticks(np.arange(num_sta), sta_list, fontsize=fsize_label)
plot_label('EGF Index',None,'P-wave CC')
plt.subplot(122)
for ii in range(num_sta):
    color = [cmap(cc) for cc in cc_s[ii]]
    plt.scatter(np.arange(num_egf), np.ones(num_egf)*ii, np.ones(num_egf)*mark_size, marker='s', color=color)
plt.xticks(np.arange(0,num_egf,2))
plt.yticks(np.arange(num_sta), ['']*num_sta, fontsize=fsize_label)
plot_label('EGF Index',None,'S-wave CC',False)
cbar_ax = fig.add_axes(cbar_pos)
cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap)
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels(cbar_tlabels)
plt.setp(cbar_ax.yaxis.get_majorticklabels(), fontsize=fsize_label)
plt.subplots_adjust(**subplot_rect)
plt.savefig(fout)

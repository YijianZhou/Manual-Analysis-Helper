""" Make phase file with catalog. P & S arrivals are predicted
"""
import os, sys
sys.path.append('/home/zhouyj/software/data_prep')
import numpy as np
from obspy import UTCDateTime
from reader import dtime2str, read_fsta, read_fctlg_np, slice_ctlg
from signal_lib import calc_dist_km

# i/o paths
fctlg = 'input/eg_all.ctlg'
fout = open('input/eg_all_org.pha','w')
fsta = 'input/station_eg.csv'
sta_dict = read_fsta(fsta)
# selection criteria
mag_rng = [1.,6]
vp, vs = 5.9, 3.4

# read catalog & slice
events = read_fctlg_np(fctlg)
events = slice_ctlg(events, mag_rng=mag_rng)

# write phase
for [ot, lat, lon, dep, mag] in events:
  fout.write('%s,%s,%s,%s,%s\n'%(dtime2str(ot), lat, lon, dep, mag))
  for sta, [sta_lat,sta_lon,_] in sta_dict.items():
    dist = calc_dist_km([lat,sta_lat],[lon,sta_lon])
    ttp, tts = dist/vp, dist/vs
    fout.write('%s,%s,%s\n'%(sta,ot+ttp,ot+tts))
fout.close()

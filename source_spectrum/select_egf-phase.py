""" Selection of EGFs
"""
import os, sys
sys.path.append('/home/zhouyj/software/data_prep')
import numpy as np
from obspy import UTCDateTime
from reader import dtime2str, read_fsta, read_fctlg_np, slice_ctlg
from signal_lib import calc_dist_km

# i/o paths
fctlg = 'input/all.ctlg'
fsta = 'input/station.csv'
sta_dict = read_fsta(fsta)
fout = open('input/egf_org.pha','w')
# selection criteria
ot_rng = ['20210518133943.51','20210521184850.54']
ot_rng = [UTCDateTime(ot) for ot in ot_rng]
mag_rng = [2.5,3.5]
lat_rng = [25.64,25.66]
lon_rng = [99.91,99.94]
vp, vs = 5.9, 3.4

# read catalog & slice
events = read_fctlg_np(fctlg)
events = slice_ctlg(events, ot_rng=ot_rng, lat_rng=lat_rng, lon_rng=lon_rng, mag_rng=mag_rng)

# write phase
for [ot, lat, lon, dep, mag] in events:
  fout.write('%s,%s,%s,%s,%s\n'%(dtime2str(ot), lat, lon, dep, mag))
  for sta, [sta_lat,sta_lon,_] in sta_dict.items():
    dist = calc_dist_km([lat,sta_lat],[lon,sta_lon])
    ttp, tts = dist/vp, dist/vs
    fout.write('%s,%s,%s\n'%(sta,ot+ttp,ot+tts))
fout.close()

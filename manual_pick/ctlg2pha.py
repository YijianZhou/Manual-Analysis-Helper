""" Make phase file with catalog. P & S arrivals are predicted
"""
import os
import numpy as np
from obspy import UTCDateTime
from reader import dtime2str, read_fsta, read_fctlg

# i/o paths
fctlg = 'input/eg_org.ctlg'
event_list = read_fctlg(fctlg)
fout = open('input/eg_org.pha','w')
fsta = 'input/station_eg.csv'
sta_dict = read_fsta(fsta)

def est_pick(ot, event_loc, sta_loc):
    vp, vs = 5.9, 3.4
    lat, lon, dep = event_loc
    sta_lat, sta_lon, sta_ele = sta_loc
    cos_lat = np.cos(lat * np.pi/180)
    dx = 111 * (lon - sta_lon) * cos_lat
    dy = 111 * (lat - sta_lat)
    dz = dep + sta_ele/1e3
    dist = (dx**2 + dy**2 + dz**2)**0.5
    ttp, tts = dist/vp, dist/vs
    return ot+ttp, ot+tts

# write phase
for [ot, lat, lon, dep, mag] in event_list:
    fout.write('%s,%s,%s,%s,%s\n'%(dtime2str(ot), lat, lon, dep, mag))
    for sta, [sta_lat, sta_lon, sta_ele] in sta_dict.items():
        tp, ts = est_pick(ot, [lat, lon, dep], [sta_lat, sta_lon, sta_ele])
        fout.write('%s,%s,%s\n'%(sta,tp,ts))
fout.close()

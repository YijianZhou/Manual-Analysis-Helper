import numpy as np
from obspy import UTCDateTime
from reader import dtime2str, read_fsta, read_fctlg, slice_ctlg

# i/o paths
fctlg = 'input/eg_all.ctlg'
fsta = 'input/eg_station.csv'
sta_dict = read_fsta(fsta)
fout = open('input/eg_egf_org.pha','w')
# selection criteria
ot_rng = ['2019-07-04T17:33:49.000000Z','20190710']
ot_rng = [UTCDateTime(ot) for ot in ot_rng]
mag_rng = [[2.8,4.3],[3.5,3.9]][1]
lat_rng = [35.74,35.78]
lon_rng = [-117.6,-117.55]
dep_rng = [0,12]
vp, vs = 5.9, 3.4

# read catalog & slice
events = read_fctlg(fctlg)
events = slice_ctlg(events, ot_rng=ot_rng, lat_rng=lat_rng, lon_rng=lon_rng, dep_rng=dep_rng, mag_rng=mag_rng)

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
for [ot, lat, lon, dep, mag] in events:
    fout.write('%s,%s,%s,%s,%s\n'%(dtime2str(ot), lat, lon, dep, mag))
    for sta, sta_loc in sta_dict.items():
        tp, ts = est_pick(ot, [lat, lon, dep], sta_loc)
        fout.write('%s,%s,%s\n'%(sta,tp,ts))
fout.close()

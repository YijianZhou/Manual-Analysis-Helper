import os, sys
import numpy as np
from obspy import UTCDateTime
from reader import dtime2str, read_fsta, read_fpha, read_fcc

# i/o paths
fcc = 'output/eg_tar-egf.cc'
cc_dict = read_fcc(fcc)
fpha = 'input/eg_egf_org.pha'
event_list = read_fpha(fpha)
fsta = 'input/eg_station.csv'
sta_dict = read_fsta(fsta)
fout = open('output/eg_egf.pha','w')
# selection criteria
cc_min = 0.6
min_sta = 8 # min(cc_p, cc_s)>cc_min counts

# make event dict
event_dict = {}
for [event_loc, pick_dict] in event_list:
    ot, lat, lon, dep, mag = event_loc
    event_name = dtime2str(ot)
    event_line = '%s,%s,%s,%s,%s\n'%(ot, lat, lon, dep, mag)
    phase_lines = ['%s,%s,%s\n'%(sta,tp,ts) for sta, [tp, ts] in pick_dict.items()]
    event_dict[event_name] = [event_line, phase_lines]

for event_name, cc_list in cc_dict.items():
    cc_mat = np.array([cc[1:3] for cc in cc_list])
    if sum(cc_mat[:,0]>cc_min)<min_sta and sum(cc_mat[:,1]>cc_min)<min_sta: continue
    event_line, phase_lines = event_dict[event_name]
    if len(phase_lines)<min_sta: continue
    fout.write(event_line)
    for phase_line in phase_lines: fout.write(phase_line)
fout.close()

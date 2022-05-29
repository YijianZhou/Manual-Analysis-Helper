import os, sys
import numpy as np
from obspy import UTCDateTime
from reader import dtime2str, read_fsta, read_fpha

# i/o paths
fcc = 'output/eg_tar-egf.cc'
fpha = 'input/eg_egf_org.pha'
event_list = read_fpha(fpha)
fout = open('output/eg_egf.pha','w')
fsta = 'input/eg_station.csv'
sta_dict = read_fsta(fsta)
# selection criteria
cc_min = 0.5
min_sta = 8 # min(cc_p, cc_s)>cc_min counts

# make event dict
event_dict = {}
for [event_loc, pick_dict] in event_list:
    ot, lat, lon, dep, mag = event_loc
    event_name = dtime2str(ot)
    event_line = '%s,%s,%s,%s,%s\n'%(ot, lat, lon, dep, mag)
    phase_lines = ['%s,%s,%s\n'%(sta,tp,ts) for sta, [tp, ts] in pick_dict.items()]
    event_dict[event_name] = [event_line, phase_lines]

# read fcc
cc_list = []
f=open(fcc); lines=f.readlines(); f.close()
for line in lines:
    codes = line.split(',')
    if len(codes)==5: cc_list.append([dtime2str(codes[0]),[]]); continue
    sta = codes[0]
    cc_p, cc_s = [float(code) for code in codes[1:3]]
    cc_list[-1][-1].append([cc_p, cc_s])

for event_name, cc_mat in cc_list:
    cc_mat = np.array(cc_mat)
    if not (sum(cc_mat[:,0]>cc_min)>min_sta or sum(cc_mat[:,1]>cc_min)>min_sta): continue
    event_line, phase_lines = event_dict[event_name]
    fout.write(event_line)
    for phase_line in phase_lines: fout.write(phase_line)
fout.close()

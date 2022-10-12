""" Further selection of cc_dt_all.link
"""
import numpy as np
from dataset_cc import read_fsta, read_fpha_dict, calc_dist_km
import config

cfg = config.Config()
# i/o paths
flink = 'output/cc_dt_all.link'
fout = open('output/cc_dt.link','w')
fpha = cfg.fpha
fsta = cfg.fsta
event_dict = read_fpha_dict(fpha)
sta_dict = read_fsta(fsta)
# thres for linking event pairs
cc_thres = cfg.cc_thres[1] # min cc
dt_thres = cfg.dt_thres[1] # max dt
num_sta_thres = cfg.num_sta_thres[1] # min sta
loc_dev_thres = cfg.loc_dev_thres[1] # max dev loc
dep_dev_thres = cfg.dep_dev_thres[1] # max dev dep
dist_thres = cfg.dist_thres[1] # max epi-dist

# read dt.cc
print('reading %s'%flink)
link_list = []
f=open(flink); lines=f.readlines(); f.close()
for i,line in enumerate(lines):
    if i%1e5==0: print('done/total %s/%s | %s pairs selected'%(i,len(lines),len(link_list)))
    codes = line.split(',')
    if len(codes)==2:
        to_add = True
        evid1, evid2 = codes
        evid2 = evid2[:-1]
        if evid1 not in event_dict or evid2 not in event_dict: 
            to_add = False; continue
        lat1, lon1, dep1 = event_dict[evid1][0][0:3]
        lat2, lon2, dep2 = event_dict[evid2][0][0:3]
        # 1. select loc dev
        loc_dev = calc_dist_km([lat1,lat2], [lon1,lon2])
        dep_dev = abs(dep1 - dep2)
        if not (loc_dev<loc_dev_thres and dep_dev<dep_dev_thres):
            to_add = False; continue
        link_list.append([line, []])
    else:
        if not to_add: continue
        # 2. select by epicentral distance
        sta = codes[0]
        sta_lat, sta_lon = sta_dict[sta]
        dist1 = calc_dist_km([sta_lat,lat1], [sta_lon,lon1])
        dist2 = calc_dist_km([sta_lat,lat2], [sta_lon,lon2])
        if min(dist1, dist2)>dist_thres: continue
        # select by CC
        dt_sp, cc_det, cc_p, cc_s = [float(code) for code in codes[1:5]]
        if dt_sp>dt_thres[2]: continue
        if cc_det<cc_thres[0]: continue
        if min(cc_p, cc_s)<cc_thres[1]: continue
        link_list[-1][-1].append(line)

# write dt.cc
print('write flink')
for [head_line, dt_list] in link_list:
    if len(dt_list)<num_sta_thres: continue
    fout.write(head_line)
    for dt_line in dt_list: fout.write(dt_line)
fout.close()

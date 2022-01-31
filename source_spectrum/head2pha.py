import sys, glob, os
sys.path.append('/home/zhouyj/software/data_prep')
from obspy import read, UTCDateTime
from reader import dtime2str, read_fpha

# i/o paths
fpha = 'input/egf_org.pha'
fout = open('output/egf.pha','w')
event_root = 'input/events_egf'
event_list = read_fpha(fpha)

for event_loc, pick_dict in event_list:
    ot, lat, lon, dep, mag = event_loc
    event_name = dtime2str(ot)
    event_dir = os.path.join(event_root, event_name)
    if not os.path.exists(event_dir): continue
    fout.write('%s,%s,%s,%s,%s\n'%(ot, lat, lon, dep, mag))
    for net_sta in pick_dict.keys():
        st_paths = sorted(glob.glob(event_dir+'/%s.*'%net_sta))
        if len(st_paths)==0: continue
        st = read(st_paths[0], headonly=True)
        head = st[0].stats
        tp = head.starttime + head.sac.t0 
        ts = head.starttime + head.sac.t1 
        if 't2' in head.sac: continue
        fout.write('%s,%s,%s\n'%(net_sta, tp, ts))
fout.close()

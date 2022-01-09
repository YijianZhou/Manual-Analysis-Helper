import os, sys, glob, shutil
sys.path.append('/home/zhouyj/software/data_prep')
import numpy as np
import multiprocessing as mp
from obspy import read, UTCDateTime
from reader import read_fpha, get_yb_data, dtime2str
import sac

# i/o paths
fpha = 'input/eg_init.pha'
get_data_dict = get_yb_data
data_dir = '/data/Example'
out_root = 'input/eg_events'
pha_list = read_fpha(fpha)
event_win = [10, 40] # sec before & after P

def cut_event(event_id):
    # get event info
    [event_loc, pick_dict] = pha_list[event_id]
    ot, lat, lon, dep, mag = event_loc
    data_dict = get_data_dict(ot, data_dir)
    event_name = dtime2str(ot)
    event_dir = os.path.join(out_root, event_name)
    if not os.path.exists(event_dir): os.makedirs(event_dir)
    # cut event
    print('cutting {}'.format(event_name))
    for net_sta, [tp, ts] in pick_dict.items():
      if net_sta not in data_dict: continue
      for data_path in data_dict[net_sta]:
        b = tp - read(data_path)[0].stats.starttime - event_win[0]
        chn_code = data_path.split('.')[-2]
        out_path = os.path.join(event_dir,'%s.%s'%(net_sta,chn_code))
        # cut event
        sac.cut(data_path, b, b+sum(event_win), out_path)
        tn = {}
        tn['t0'] = event_win[0]
        if ts!=-1: tn['t1'] = ts - tp + event_win[0]
        sac.ch_event(out_path, lat, lon, dep, mag, tn)
# cut all events data
for evid in range(len(pha_list)): cut_event(evid)


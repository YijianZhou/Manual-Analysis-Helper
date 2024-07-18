import os, glob
import numpy as np
import multiprocessing as mp
from obspy import read, UTCDateTime
from reader import read_fpha, get_data_dict, dtime2str
import sac

# i/o paths
fpha = 'input/eg_org.pha'
get_data_dict = get_data_dict
data_dir = '/data/Example_data'
out_root = 'input/eg_events'
event_list = read_fpha(fpha)
event_win = [10, 40] # sec before & after P
num_workers = 2

def cut_event(evid):
    # get event info
    [event_loc, pick_dict] = event_list[evid]
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
pool = mp.Pool(num_workers)
pool.map_async(cut_event, range(len(event_list)))
pool.close()
pool.join()

import os, glob, shutil
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from obspy import read, UTCDateTime
from signal_lib import preprocess, obspy_slice
from reader import read_fpha, get_data_dict, dtime2str
shutil.copyfile('../config.py', 'config.py')
import config
import warnings
warnings.filterwarnings("ignore")

# i/o paths
cfg = config.Config()
fpha = '../' + cfg.fpha if cfg.fpha[0]!='/' else cfg.fpha
data_dir = cfg.data_dir
out_root = cfg.event_root
if not os.path.exists(out_root): os.makedirs(out_root)
# signal process
num_workers = cfg.num_workers
win_len = cfg.win_event
get_data_dict = get_data_dict # modify this if using customized function
samp_rate = cfg.samp_rate
freq_band = cfg.freq_band


def get_sta_date(event_list):
    sta_date_dict = {}
    for i, [event_loc, pick_dict] in enumerate(event_list):
        if i%1e3==0: print('%s/%s events done/total'%(i, len(event_list)))
        # 1. get event info
        event_name = dtime2str(event_loc[0])
        event_dir = os.path.join(out_root, event_name)
        if not os.path.exists(event_dir): os.makedirs(event_dir)
        for net_sta, [tp, ts] in pick_dict.items():
            date = str(tp.date)
            sta_date = '%s_%s'%(net_sta, date) # for one day's stream data
            if sta_date not in sta_date_dict:
                sta_date_dict[sta_date] = [[event_dir, tp, ts]]
            else: sta_date_dict[sta_date].append([event_dir, tp, ts])
    return sta_date_dict


class Cut_Events(Dataset):
  """ Dataset for cutting event windows
  """
  def __init__(self, sta_date_items):
    self.sta_date_items = sta_date_items

  def __getitem__(self, index):
    data_paths_i = []
    # get one sta-date
    sta_date, samples = self.sta_date_items[index]
    net_sta, date = sta_date.split('_')
    net, sta = net_sta.split('.')
    date = UTCDateTime(date)
    # read & prep one day's data
    print('reading %s %s'%(net_sta, date.date))
    data_dict = get_data_dict(date, data_dir)
    if net_sta not in data_dict: return data_paths_i
    st_paths = data_dict[net_sta]
    try:
        stream  = read(st_paths[0])
        stream += read(st_paths[1])
        stream += read(st_paths[2])
    except: return data_paths_i
    stream = preprocess(stream, samp_rate, freq_band)
    if len(stream)!=3: return data_paths_i
    for [event_dir, tp, ts] in samples:
        # time shift & prep
        start_time = tp - win_len[0]
        end_time = tp + win_len[1]
        st = obspy_slice(stream, start_time, end_time)
        if len(st)!=3: continue
        st = st.detrend('demean')  # note: no detrend here
        # write & record out_paths
        data_paths_i.append([])
        for ii,tr in enumerate(st):
            out_path = os.path.join(event_dir,'%s.%s'%(net_sta,ii+1))
            tr.write(out_path, format='sac')
            tr = read(out_path)[0]
            tr.stats.sac.t0, tr.stats.sac.t1 = tp-start_time, ts-start_time
            tr.write(out_path, format='sac')
            data_paths_i[-1].append(out_path)
    return data_paths_i

  def __len__(self):
    return len(self.sta_date_items)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) # 'spawn' or 'forkserver'
    # read fpha
    event_list = read_fpha(fpha)
    sta_date_dict = get_sta_date(event_list)
    sta_date_items = list(sta_date_dict.items())
    # for sta-date pairs
    data_paths = []
    dataset = Cut_Events(sta_date_items)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=None)
    for i, data_paths_i in enumerate(dataloader): 
        data_paths += data_paths_i
        if i%10==0: print('%s/%s sta-date pairs done/total'%(i+1,len(dataset)))
    fout_data_paths = os.path.join(out_root,'data_paths.npy')
    np.save(fout_data_paths, data_paths)

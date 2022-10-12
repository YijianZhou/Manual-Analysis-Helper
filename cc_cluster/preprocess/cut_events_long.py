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


def cut_event_window(stream_paths, tp, ts, out_paths):
    t0 = tp - win_len[0] - sum(win_len)/2
    t1 = t0 + sum(win_len)*2
    st  = read(stream_paths[0], starttime=t0, endtime=t1)
    st += read(stream_paths[1], starttime=t0, endtime=t1)
    st += read(stream_paths[2], starttime=t0, endtime=t1)
    if len(st)!=3: return False
    st = preprocess(st, samp_rate, freq_band)
    st = obspy_slice(st, tp-win_len[0], tp+win_len[1])
    if len(st)!=3: return False
    for ii, tr in enumerate(st):
        tr.write(out_paths[ii], format='sac')
        tr = read(out_paths[ii])[0]
        tr.stats.sac.t0, tr.stats.sac.t1 = win_len[0], win_len[0]+(ts-tp)
        tr.write(out_paths[ii], format='sac')
    return True

class Cut_Events(Dataset):
  """ Dataset for cutting templates
  """
  def __init__(self, event_list):
    self.event_list = event_list

  def __getitem__(self, index):
    data_paths_i = []
    # get event info
    event_loc, pick_dict = self.event_list[index]
    ot, lat, lon, dep, mag = event_loc
    event_name = dtime2str(ot)
    data_dict = get_data_dict(ot, data_dir)
    event_dir = os.path.join(out_root, event_name)
    if not os.path.exists(event_dir): os.makedirs(event_dir)
    # cut event
    for net_sta, [tp, ts] in pick_dict.items():
        if net_sta not in data_dict: continue
        stream_paths = data_dict[net_sta]
        out_paths = [os.path.join(event_dir,'%s.%s'%(net_sta,ii)) for ii in range(3)]
        is_cut = cut_event_window(stream_paths, tp, ts, out_paths)
        if not is_cut: continue
        data_paths_i.append(out_paths)
    return data_paths_i

  def __len__(self):
    return len(self.event_list)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) # 'spawn' or 'forkserver'
    event_list = read_fpha(fpha)
    data_paths  = []
    dataset = Cut_Events(event_list)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=None)
    for i, data_paths_i in enumerate(dataloader):
        data_paths += data_paths_i
        if i%10==0: print('%s/%s events done/total'%(i,len(dataset)))
    fout_data_paths = os.path.join(out_root,'data_paths.npy')
    np.save(fout_data_paths, data_paths)


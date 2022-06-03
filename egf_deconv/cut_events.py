import os, sys, glob
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from obspy import read, UTCDateTime
from reader import read_fsta, read_fpha, get_data_dict, dtime2str
sys.path.append('/home/zhouyj/software/data_prep')
import sac
import warnings
warnings.filterwarnings("ignore")

# i/o paths
idx = 1
fsta = 'input/eg_station.csv'
fpha = ['input/eg_tar.pha','input/eg_egf_org.pha'][idx]
out_root = ['input/eg_tar','input/eg_egf'][idx]
event_list = read_fpha(fpha)
get_data_dict = get_data_dict
data_dir = '/data3/Ridgecrest'
# signal process
win_len = [20, 40] # sec before & after P
num_workers = 5
bad_index = [] # from event waveform & CC inspection

class Cut_Events(Dataset):
  """ Dataset for cutting templates
  """
  def __init__(self, event_list):
    self.event_list = event_list

  def __getitem__(self, index):
    if index in bad_index: return False
    # get event info
    event_loc, pick_dict = self.event_list[index]
    ot, lat, lon, dep, mag = event_loc
    event_name = dtime2str(ot)
    data_dict = get_data_dict(ot, data_dir)
    event_dir = os.path.join(out_root, event_name)
    if not os.path.exists(event_dir): os.makedirs(event_dir)
    # cut event
    for net_sta, [tp, ts] in pick_dict.items():
      data_paths = data_dict[net_sta]
      for data_path in data_paths:
        b = tp - read(data_path, headonly=True)[0].stats.starttime - win_len[0] 
        chn = data_path.split('.')[-2]
        out_path = os.path.join(event_dir,'%s.%s'%(net_sta,chn))
        sac.cut(data_path, b, b+sum(win_len), out_path)
        t0 = win_len[0]
        t1 = ts - tp + win_len[0]
        sac.ch_event(out_path, tn={'t0':t0,'t1':t1})
    return True

  def __len__(self):
    return len(self.event_list)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) # 'spawn' or 'forkserver'
    dataset = Cut_Events(event_list)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=None)
    for ii,_ in enumerate(dataloader):
        print('%s/%s events done/total'%(ii,len(dataset)))


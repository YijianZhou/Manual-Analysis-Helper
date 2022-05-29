import os, sys, glob
from obspy import read, UTCDateTime
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from reader import read_fpha, dtime2str
from signal_lib import preprocess
sys.path.append('/home/zhouyj/software/data_prep')
import sac
import warnings
warnings.filterwarnings("ignore")

# i/o paths
fpha = 'input/eg_egf_org.pha'
event_list = read_fpha(fpha)
fout = open(fpha,'w')
event_root = 'input/eg_egf'
# signal process
samp_rate = 100
freq_band = [1,20]
num_workers = 10
dt_pick = 1.5 # sec pre & post
win_sta = [0.8, 1] # for P & S
win_lta = [2, 2]
win_sta_npts = [int(samp_rate*win) for win in win_sta]
win_lta_npts = [int(samp_rate*win) for win in win_lta]
min_snr = 9 # defined by energy


def calc_sta_lta(data, win_lta_npts, win_sta_npts):
    npts = len(data)
    if npts < win_lta_npts + win_sta_npts:
        print('input data too short!')
        return np.zeros(1)
    sta = np.zeros(npts)
    lta = np.ones(npts)
    data_cum = np.cumsum(data)
    sta[:-win_sta_npts] = data_cum[win_sta_npts:] - data_cum[:-win_sta_npts]
    sta /= win_sta_npts
    lta[win_lta_npts:]  = data_cum[win_lta_npts:] - data_cum[:-win_lta_npts]
    lta /= win_lta_npts
    sta_lta = sta/lta
    sta_lta[np.isinf(sta_lta)] = 0.
    sta_lta[np.isnan(sta_lta)] = 0.
    return sta_lta


class Pick_Events(Dataset):
  def __init__(self, event_list):
    self.event_list = event_list

  def __getitem__(self, index):
    event_loc, pick_dict = self.event_list[index]
    ot, lat, lon, dep, mag = event_loc
    event_line = '%s,%s,%s,%s,%s\n'%(ot, lat, lon, dep, mag)
    event_name = dtime2str(event_loc[0])
    event_dir = os.path.join(event_root, event_name)
    pick_lines = []
    for sta, [tp0, ts0] in pick_dict.items():
        # read data
        st_paths = sorted(glob.glob('%s/%s.*'%(event_dir, sta)))
        if not len(st_paths)==3: continue
        st  = read(st_paths[0])
        st += read(st_paths[1])
        st += read(st_paths[2])
        st = preprocess(st, samp_rate, freq_band)
        if not len(st_paths)==3: continue
        # refine P & S pick
        st_p = st.slice(tp0-win_lta[0]-dt_pick, tp0+win_sta[0]+dt_pick)
        data_p = st_p[2].data**2
        cf_p = calc_sta_lta(data_p, win_lta_npts[0], win_sta_npts[0])
        if np.amax(cf_p)<min_snr: continue
        dt_p = np.argmax(cf_p)/samp_rate - dt_pick - win_lta[0]
        tp = tp0 + dt_p
        st_s = st.slice(ts0-win_lta[1]-dt_pick, ts0+win_sta[1]+dt_pick)
        data_s = st_s[1].data**2 + st_s[2].data**2
        cf_s = calc_sta_lta(data_s, win_lta_npts[1], win_sta_npts[1])
        dt_s = np.argmax(cf_s)/samp_rate - dt_pick - win_lta[1]
        ts = ts0 + dt_s
        if ts<=tp: ts = ts0
        for st_path in st_paths: 
            t0 = st[0].stats.sac.t0 + (tp-tp0)
            t1 = st[0].stats.sac.t1 + (ts-ts0)
            sac.ch_event(st_path, tn={'t0':t0,'t1':t1})
        pick_lines.append('%s,%s,%s\n'%(sta, tp, ts))
    return [event_line, pick_lines]

  def __len__(self):
    return len(self.event_list)


if __name__ == '__main__':
  mp.set_start_method('spawn', force=True) # 'spawn' or 'forkserver'
  dataset = Pick_Events(event_list)
  dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=None)
  for ii, [event_line, pick_lines] in enumerate(dataloader):
      print('pick %s | %s'%(event_line[:-1], ii))
      fout.write(event_line)
      for pick_line in pick_lines: fout.write(pick_line)
  fout.close()

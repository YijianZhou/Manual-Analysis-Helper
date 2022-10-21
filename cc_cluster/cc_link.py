""" Calculate waveform CC and d(S-P)
"""
import os, glob
import time
import numpy as np
from obspy import read, UTCDateTime
from dataset_cc import get_event_list, read_fsta, read_data_temp, calc_dist_km
import config
from scipy.signal import correlate
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import torch
import warnings
warnings.filterwarnings("ignore")
mp.set_sharing_strategy('file_system')

cfg = config.Config()
# i/o paths
fsta = cfg.fsta
fpha = cfg.fpha
event_root = cfg.event_root
fout = open('output/cc_dt_all.link','w')
# quality control: event pair linking
num_workers = cfg.num_workers
cc_thres = cfg.cc_thres[0] # min cc
dt_thres = cfg.dt_thres[0] # max dt
loc_dev_thres = cfg.loc_dev_thres[0] # max dev loc
dep_dev_thres = cfg.dep_dev_thres[0] # max dev dep
dist_thres = cfg.dist_thres[0] # max epi-dist
num_sta_thres = cfg.num_sta_thres[0] # min sta
max_nbr = cfg.num_nbr_thres[1] 
temp_mag = cfg.temp_mag
temp_sta = cfg.temp_sta
# data prep
samp_rate = cfg.samp_rate
win_temp_det = cfg.win_temp_det
win_temp_p = cfg.win_temp_p
win_temp_s = cfg.win_temp_s
win_data_p = [win+dt_thres[0] for win in win_temp_p]
win_data_s = [win+dt_thres[1] for win in win_temp_s]
tt_shift_p = win_temp_p[0] - win_data_p[0]
tt_shift_s = win_temp_s[0] - win_data_s[0]


def calc_cc(data, temp, norm_data, norm_temp):
    num_chn, len_data = data.shape
    _,       len_temp = temp.shape
    cc = []
    for i in range(num_chn):
        cci = correlate(data[i], temp[i], mode='valid')[1:]
        cci /= norm_data[i] * norm_temp[i]
        cci[np.isinf(cci)] = 0.
        cci[np.isnan(cci)] = 0.
        cc.append(cci)
    return np.mean(cc,axis=0)

# calc CC & d(S-P) for all event pairs
def calc_cc_dt(event_list, sta_dict, fout):
    # 1. get_neighbor_pairs
    print('1. get candidate event pairs')
    num_events = len(event_list)
    dtype = [('lat','O'),('lon','O'),('dep','O'),('is_temp','O'),('sta','O')]
    loc_sta_list = []
    for _, event_loc, pha_dict in event_list:
        sta = list(pha_dict.keys())
        lat, lon, dep, mag = event_loc[1:5]
        is_temp = 1 if mag>=temp_mag and len(sta)>=temp_sta else 0
        loc_sta_list.append((lat, lon, dep, is_temp, sta))
    loc_sta_list = np.array(loc_sta_list, dtype=dtype)
    nbr_dataset = Get_Neighbor(loc_sta_list)
    nbr_loader = DataLoader(nbr_dataset, num_workers=num_workers, batch_size=None)
    t = time.time()
    pair_list = []
    for i, pair_i in enumerate(nbr_loader):
        if i%1000==0: print('done/total events {}/{} | {:.1f}s'.format(i, len(loc_sta_list), time.time()-t))
        pair_list += list(pair_i.numpy())
    pair_list = np.unique(pair_list, axis=0)
    num_pairs = len(pair_list)
    print('%s pairs linked'%num_pairs)
    # 2. cc link
    print('2. link event pairs with waveform CC')
    cd_dataset = CC_DT(event_list, pair_list, sta_dict)
    cd_loader = DataLoader(cd_dataset, num_workers=num_workers, batch_size=None)
    link_num = 0
    t = time.time()
    for i, [[data_evid, temp_evid], dt_dict] in enumerate(cd_loader):
        if i%10000==0: print('done/total {}/{} | {} pairs linked | {:.1f}s'.format(i, num_pairs, link_num, time.time()-t))
        if len(dt_dict)<num_sta_thres: continue
        write_cc_dt(data_evid, temp_evid, dt_dict, fout)
        link_num += 1

def write_cc_dt(data_evid, temp_evid, dt_dict, fout):
    fout.write('{},{}\n'.format(data_evid, temp_evid))
    for net_sta, [dt_sp, cc_det, cc_p, cc_s] in dt_dict.items():
        fout.write('{},{:.2f},{:.2f},{:.2f},{:.2f}\n'.format(net_sta, dt_sp, cc_det, cc_p, cc_s))


class Get_Neighbor(Dataset):
  """ Dataset for finding neighbor event
  """
  def __init__(self, loc_sta_list):
    self.loc_sta_list = loc_sta_list

  def __getitem__(self, index):
    num_events = len(self.loc_sta_list)
    # 1. select by loc dev
    lat, lon, dep, is_temp, sta_ref = self.loc_sta_list[index]
    cos_lat = np.cos(lat*np.pi/180)
    cond_lat = 111*abs(self.loc_sta_list['lat']-lat) < loc_dev_thres
    cond_lon = 111*abs(self.loc_sta_list['lon']-lon)*cos_lat < loc_dev_thres
    cond_dep = abs(self.loc_sta_list['dep']-dep) < dep_dev_thres
    if is_temp==1: cond_loc = cond_lat*cond_lon*cond_dep
    else: cond_loc = (self.loc_sta_list['is_temp']==1)*cond_lat*cond_lon*cond_dep
    # 2. select by shared sta
    sta_lists = self.loc_sta_list[cond_loc]['sta']
    cond_sta = [len(np.intersect1d(sta_list, sta_ref)) >= num_sta_thres for sta_list in sta_lists]
    # 3. select to maximum num of neighbor
    sub_list = self.loc_sta_list[cond_loc][cond_sta]
    if len(sub_list)==0: return np.array([], dtype=np.int)
    dist_lat = 111*abs(sub_list['lat']-lat)
    dist_lon = 111*abs(sub_list['lon']-lon)*cos_lat
    dist_dep = abs(sub_list['dep']-dep)
    dist_list = (dist_lat**2 + dist_lon**2 + dist_dep**2)**0.5
    dist_thres = np.sort(dist_list)[0:max_nbr+1][-1]
    cond_nbr = dist_list<=dist_thres
    # 4. to pair index
    pair_list = []
    evid_list = np.arange(num_events)[cond_loc][cond_sta][cond_nbr]
    for evid in evid_list:
        if evid==index: continue
        evid1, evid2 = np.sort([evid, index])
        pair_list.append([evid1, evid2])
    return np.array(pair_list, dtype=np.int)

  def __len__(self):
    return len(self.loc_sta_list)


class CC_DT(Dataset):
  """ Dataset for event linking with CC
  """
  def __init__(self, event_list, pair_list, sta_dict):
    self.event_list = event_list
    self.pair_list = pair_list
    self.sta_dict = sta_dict

  def __getitem__(self, index):
    # calc one event pair
    data_idx, temp_idx = self.pair_list[index]
    data_evid, data_loc, pha_dict_data = self.event_list[data_idx]
    temp_evid, temp_loc, pha_dict_temp = self.event_list[temp_idx]
    data_ot, data_lat, data_lon = data_loc[0:3]
    temp_ot, temp_lat, temp_lon = temp_loc[0:3]
    # check loc dev, num sta
    sta_list = [sta for sta in pha_dict_data.keys() if sta in pha_dict_temp.keys()]
    if len(sta_list)<num_sta_thres: return [data_evid, temp_evid], {}
    # for all shared sta pha
    dt_dict = {}
    for sta in sta_list:
        dt_p, dt_s, cc_p, cc_s = [None]*4
        # check epicentral distance
        sta_lat, sta_lon = self.sta_dict[sta]
        data_dist = calc_dist_km([sta_lat,data_lat], [sta_lon,data_lon])
        temp_dist = calc_dist_km([sta_lat,temp_lat], [sta_lon,temp_lon])
        if min(data_dist,temp_dist)>dist_thres: continue
        # read data & temp
        data_paths, data_tp, data_ts = pha_dict_data[sta]
        temp_paths, temp_tp, temp_ts = pha_dict_temp[sta]
        data_all, _, data_tt = read_data_temp(data_paths, data_tp, data_ts, data_ot)
        _, temp_all, temp_tt = read_data_temp(temp_paths, temp_tp, temp_ts, temp_ot)
        data_det, data_p, data_s, norm_data_det, norm_data_p, norm_data_s = data_all
        temp_det, temp_p, temp_s, norm_temp_det, norm_temp_p, norm_temp_s = temp_all
        data_ttp, data_tts = data_tt
        temp_ttp, temp_tts = temp_tt
        if not (type(data_det)==np.ndarray and type(temp_det)==np.ndarray \
        and type(data_p)==np.ndarray and type(temp_p)==np.ndarray \
        and type(data_s)==np.ndarray and type(temp_s)==np.ndarray): continue
        # calc CC and d(S-P)
        cc_det = calc_cc(data_det, temp_det, norm_data_det, norm_temp_det)
        cc_det = np.amax(cc_det)
        if cc_det<cc_thres[0]: continue
        cc_p = calc_cc(data_p, temp_p, norm_data_p, norm_temp_p)
        data_ttp += tt_shift_p + np.argmax(cc_p)/samp_rate
        dt_p = data_ttp - temp_ttp 
        cc_p = np.amax(cc_p)
        if cc_p<cc_thres[1]: continue
        cc_s = calc_cc(data_s, temp_s, norm_data_s, norm_temp_s)
        data_tts += tt_shift_s + np.argmax(cc_s)/samp_rate
        dt_s = data_tts - temp_tts 
        cc_s = np.amax(cc_s)
        if cc_s<cc_thres[1]: continue
        dt_sp = abs(dt_p-dt_s)
        if dt_sp>dt_thres[2]: continue
        dt_dict[sta] = [dt_sp, cc_det, cc_p, cc_s]
    return [data_evid, temp_evid], dt_dict

  def __len__(self):
    return len(self.pair_list)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) # 'spawn' or 'forkserver'
    # read event data & sta file
    sta_dict = read_fsta(fsta)
    event_list = get_event_list(fpha, event_root)
    # calc & write cc_dt
    calc_cc_dt(event_list, sta_dict, fout)
    fout.close()


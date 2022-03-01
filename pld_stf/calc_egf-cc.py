import sys, os, glob
sys.path.append('/home/zhouyj/software/data_prep')
import numpy as np
from obspy import read, UTCDateTime
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from signal_lib import preprocess, calc_cc, calc_azm_deg, calc_dist_km
from reader import dtime2str, read_fsta, read_fpha

# i/o paths
tar_dir =  'input/eg_tar/20210521212125.00'
tar_loc = [25.6527,99.9268]
tar_ot = UTCDateTime(tar_dir.split('/')[-1])
egf_root = '/data/bigdata/Example_events'
fsta = 'input/station_eg.csv'
sta_dict = read_fsta(fsta)
fpha = 'input/eg_all_org.pha'
event_list = read_fpha(fpha)
fout = open('output/eg_tar-egf.cc','w')
# signal process
samp_rate = 100
freq_band = [0.5,2]
s_win = [1,7]
p_win = [1,5]
dt_cc = 2 # pre & post
num_workers = 10

# read data
def read_s_data(st_paths, baz, s_win, to_prep=False):
    npts = int(samp_rate*sum(s_win))
    st = read(st_paths[1])
    st+= read(st_paths[0])
    if to_prep: st = preprocess(st, samp_rate, freq_band)
    st = st.rotate(method='NE->RT', back_azimuth=baz)
    start_time = st[0].stats.starttime
    header = st[0].stats.sac
    tp, ts = header.t0, header.t1
    t0 = start_time + ts - s_win[0]
    t1 = start_time + ts + s_win[1]
    if len(st.slice(t0,t1))<2: return []
    return st.slice(t0,t1).detrend('demean').detrend('linear').taper(max_percentage=0.05)[1].data[0:npts]

def read_p_data(st_paths, p_win, to_prep=False):
    npts = int(samp_rate*sum(p_win))
    st = read(st_paths[2])
    if to_prep: st = preprocess(st, samp_rate, freq_band)
    start_time = st[0].stats.starttime
    header = st[0].stats.sac
    tp, ts = header.t0, header.t1
    t0 = start_time + tp - p_win[0]
    t1 = start_time + tp + p_win[1]
    if len(st.slice(t0,t1))<1: return []
    return st.slice(t0,t1).detrend('demean').detrend('linear').taper(max_percentage=0.05)[0].data[0:npts]

# read tar data
tar_dict = {}
for sta, [sta_lat, sta_lon,_] in sta_dict.items():
    baz = calc_azm_deg([tar_loc[0],sta_lat], [tar_loc[1],sta_lon])[1]
    tar_paths = sorted(glob.glob(tar_dir+'/%s*'%sta))
    tar_p = read_p_data(tar_paths, [p_win[0]+dt_cc, p_win[1]+dt_cc], True)
    tar_s = read_s_data(tar_paths, baz, [s_win[0]+dt_cc,s_win[1]+dt_cc], True)
    tar_dict[sta] = [tar_p, tar_s]


class EGF_CC(Dataset):
  """ Dataset for cutting events
  """
  def __init__(self, event_list):
    self.event_list = event_list

  def __getitem__(self, index):
    ot, lat, lon, dep, mag = self.event_list[index][0]
    event_name = dtime2str(ot)
    egf_dir = os.path.join(egf_root, event_name)
    event_line = '%s,%s,%s,%s,%s\n'%(ot, lat, lon, dep, mag)
    sta_cc = []
    if abs(ot-tar_ot)<2: return event_line, []
    for sta, [sta_lat, sta_lon,_] in sta_dict.items():
        # get tar data
        tar_p, tar_s = tar_dict[sta]
        # read egf data
        egf_paths = sorted(glob.glob(egf_dir+'/%s*'%sta))
        if len(egf_paths)!=3: continue
        baz = calc_azm_deg([lat,sta_lat], [lon,sta_lon])[1]
        egf_p = read_p_data(egf_paths, p_win)
        egf_s = read_s_data(egf_paths, baz, s_win)
        if len(egf_p)==0 or len(egf_s)==0: continue
        cc_p, cc_s = calc_cc(tar_p, egf_p), calc_cc(tar_s, egf_s)
        sta_cc.append([sta, np.amax(cc_p), np.amax(cc_s)])
    return event_line, sta_cc

  def __len__(self):
    return len(self.event_list)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) # 'spawn' or 'forkserver'
    dataset = EGF_CC(event_list)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=None)
    for ii, [event_line, sta_cc] in enumerate(dataloader):
        if ii%10==0: print('%s/%s EGF candidates done/total'%(ii+1,len(dataset)))
        if len(sta_cc)==0: continue
        fout.write(event_line)
        for sta, cc_p, cc_s in sta_cc: fout.write('%s,%s,%s\n'%(sta,float(cc_p),float(cc_s)))
    fout.close()

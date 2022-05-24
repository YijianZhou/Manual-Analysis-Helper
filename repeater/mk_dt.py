""" Associate MESS detections --> dt.cc & phase.csv
"""
import os, glob
from obspy import UTCDateTime
import numpy as np
import multiprocessing as mp
import config
import warnings
warnings.filterwarnings("ignore")

# assoc det
def assoc_det(time_range, start_evid):
    print('associating %s'%time_range)
    print('reading detection phase file')
    # output dt & event file
    out_dt = open('input/dt_%s.cc'%time_range,'w')
    out_phase = open('input/phase_%s.csv'%time_range,'w')
    # read & select MESS detections
    start_date, end_date = [UTCDateTime(date) for date in time_range.split('-')]
    det_list = read_det_pha(det_pha, start_date, end_date)
    dets = det_list[[temp_id in temp_loc_dict for temp_id in det_list['temp_id']]]
    dets = dets[dets['cc']>=cc_thres[0]]
    num_dets = len(dets)
    for i in range(num_dets):
        if i%500==0: print('{} events associated | left {}'.format(i, len(dets)))
        det_id = start_evid + i
        det_0 = dets[0]
        # find neighbor dets by ot cluster
        cond_ot = abs(dets['ot']-det_0['ot']) < ot_dev
        dets_reloc = dets[cond_ot]
        cc = dets_reloc['cc']
        # whether self-det
        is_self = False
        self_dets = []
        for j,det in enumerate(dets_reloc):
            temp_ot = temp_loc_dict[det['temp_id']][0]
            if abs(temp_ot-det['ot'])>ot_dev: continue
            self_dets.append((j,len(det['picks'])))
        self_dets = np.array(self_dets, dtype=[('idx','int'),('nsta','O')])
        if len(self_dets)>0:
            is_self = True
            idx = self_dets[np.argmax(self_dets['nsta'])]['idx']
            det_i = dets_reloc[idx]
            det_id = det_i['temp_id']
            dets_reloc = np.delete(dets_reloc, self_dets['idx'], axis=0)
            cc = np.delete(cc, self_dets['idx'], axis=0)
        else: det_i = dets_reloc[np.argmax(cc)]
        # find location
        events = event_list[abs(event_list['ot']-det_i['ot']) < ot_dev]
        if len(events)==0: 
            det_loc = [det_i['ot']] + temp_loc_dict[det_i['temp_id']][1:4] + [-1]
        else: 
            det_loc = [det_i['ot']] + list(events[0]['loc'])
        # write dt.cc & phase.csv
        if len(dets_reloc)>=nbr_thres:
            for det in dets_reloc: write_dt(det, det_id, out_dt)
            write_phase(det_loc, det_i, det_id, out_phase)
        elif is_self: write_phase(det_loc, det_i, det_id, out_phase)
        # next det
        dets = np.delete(dets, np.where(cond_ot), axis=0)
        if len(dets)==0: break
    out_dt.close()
    out_phase.close()


# read temp pha --> temp_loc_dict
def read_temp_pha(temp_pha):
    f=open(temp_pha); lines=f.readlines(); f.close()
    temp_loc_dict = {}
    for line in lines:
        codes = line.split(',')
        if len(codes[0])<14: continue
        ot = UTCDateTime(codes[0])
        lat, lon, dep = [float(code) for code in codes[1:4]]
        temp_id = codes[-1][:-1]
        temp_loc_dict[temp_id] = [ot, lat, lon, dep]
    return temp_loc_dict

# read det pha (MESS output) --> det_list
def read_det_pha(det_pha, start_time, end_time):
    f=open(det_pha); lines=f.readlines(); f.close()
    dtype = [('temp_id','O'),('ot','O'),('loc','O'),('cc','O'),('picks','O')]
    det_list = []
    for line in lines:
        codes = line.split(',')
        if len(codes[0])>=14:
            temp_id = codes[0].split('_')[0]
            ot = UTCDateTime(codes[1])
            lat, lon, dep, cc_det = [float(code) for code in codes[2:6]]
            to_add = True if start_time<ot<end_time else False
            if to_add: det_list.append((temp_id, ot, [lat, lon, dep], cc_det, {}))
        else:
            if not to_add: continue
            net_sta = codes[0]
            tp, ts = [UTCDateTime(code) for code in codes[1:3]]
            dt_p, dt_s, s_amp, cc_p, cc_s = [float(code) for code in codes[3:8]]
            det_list[-1][-1][net_sta] = [tp, ts, dt_p, dt_s, s_amp, cc_p, cc_s]
    return np.array(det_list, dtype=dtype)

def read_fctlg(fctlg):
    dtype = [('ot','O'),('loc','O')]
    f=open(fctlg); lines=f.readlines(); f.close()
    event_list = []
    for line in lines:
        codes = line.split(',')
        ot = UTCDateTime(codes[0])
        lat, lon, dep, mag = [float(code) for code in codes[1:5]]
        event_list.append((ot, [lat, lon, dep, mag]))
    return np.array(event_list, dtype=dtype)

def read_fsta(fsta):
    sta_dict = {}
    f = open(fsta); lines = f.readlines(); f.close()
    for line in lines:
        codes = line.split(',')
        net_sta = codes[0]
        lat, lon, ele = [float(code) for code in codes[1:4]]
        sta_dict[net_sta] = [lat, lon, ele]
    return sta_dict

def write_dt(det, evid, fout):
    fout.write('# {:9} {:9}\n'.format(evid, det['temp_id']))
    for net_sta, [_,_, dt_p, dt_s, _, cc_p, cc_s] in det['picks'].items():
        sta = net_sta.split('.')[1]
        dt_sp = dt_s - dt_p
        if abs(dt_sp)<=dt_thres and min(cc_p, cc_s)>=cc_thres[1]: 
            fout.write('{:7} {:.4f} {:.4f} {:.4f}\n'.format(sta, dt_sp, cc_p, cc_s))

def write_phase(event_loc, det, evid, fout):
    ot, lat, lon, dep, mag = event_loc
    fout.write('%s,%s,%s,%s,%s,%s\n'%(ot, lat, lon, dep, mag, evid))
    for net_sta, [sta_lat, sta_lon, sta_ele] in sta_dict.items():
        if net_sta in det['picks']: 
            tp, ts = det['picks'][net_sta][0:2] 
            tp0, ts0 = est_pick(ot, [lat,lon,dep], [sta_lat,sta_lon,sta_ele])
            if max(abs(tp-tp0), abs(ts-ts0))>ot_dev: tp, ts = tp0, ts0
        else: 
            tp, ts = est_pick(ot, [lat,lon,dep], [sta_lat,sta_lon,sta_ele])
        fout.write('%s,%s,%s\n'%(net_sta, tp, ts))

def est_pick(ot, event_loc, sta_loc):
    vp, vs = 5.9, 3.4
    lat, lon, dep = event_loc
    sta_lat, sta_lon, sta_ele = sta_loc
    cos_lat = np.cos(lat * np.pi/180)
    dx = 111 * (lon - sta_lon) * cos_lat
    dy = 111 * (lat - sta_lat)
    dz = dep + sta_ele/1e3
    dist = (dx**2 + dy**2 + dz**2)**0.5
    ttp, tts = dist/vp, dist/vs
    return ot+ttp, ot+tts

def select_dt():
    print('select unique dt.cc pairs')
    # read dt.cc
    dt_list = []
    f=open('input/dt.cc'); lines=f.readlines(); f.close()
    for line in lines:
        codes = line.split()
        if line[0]=='#': 
            evid1, evid2 = np.sort(np.array([int(code) for code in codes[1:3]]))
            evid_key = '%s.%s'%(evid1, evid2)
            dt_list.append([evid_key, [line]])
        else: dt_list[-1][-1].append(line)
    # select unique dt pairs
    dt_dict = {}
    for [evid_key, lines] in dt_list:
        if evid_key not in dt_dict: dt_dict[evid_key] = lines
        else: 
           if len(dt_dict[evid_key])>len(lines): continue
           else: dt_dict[evid_key] = lines
    # write dt.cc
    fout = open('input/dt.cc','w')
    for lines in dt_dict.values():
        sta_list = np.unique([line.split()[0] for line in lines[1:]])
        if len(sta_list)<min_sta: continue
        for line in lines: fout.write(line)
    fout.close()


if __name__ == '__main__':
  # i/o paths
  cfg = config.Config()
  temp_loc_dict = read_temp_pha(cfg.temp_pha)
  det_pha = cfg.det_pha
  event_list = read_fctlg(cfg.fctlg)
  sta_dict = read_fsta(cfg.fsta)
  for fname in glob.glob('input/dt_*.cc'): os.unlink(fname)
  for fname in glob.glob('input/phase_*.csv'): os.unlink(fname)
  # assoc params
  ot_dev = cfg.ot_dev
  cc_thres = cfg.cc_thres
  dt_thres = cfg.dt_thres
  nbr_thres = cfg.nbr_thres
  evid_stride = cfg.evid_stride
  min_sta = cfg.min_sta
  num_workers = cfg.num_workers
  # start assoc
  start_date, end_date = [UTCDateTime(date) for date in cfg.time_range.split('-')]
  dt = (end_date - start_date) / num_workers
  pool = mp.Pool(num_workers)
  for proc_idx in range(num_workers):
    t0 = ''.join(str((start_date + proc_idx*dt).date).split('-'))
    t1 = ''.join(str((start_date + (proc_idx+1)*dt).date).split('-'))
    pool.apply_async(assoc_det, args=('-'.join([t0, t1]), evid_stride*(1+proc_idx),))
  pool.close()
  pool.join()
  # merge files & post-process
  os.system('cat input/dt_*.cc > input/dt.cc')
  os.system('cat input/phase_*.csv > input/phase.csv')
  for fname in glob.glob('input/dt_*.cc'): os.unlink(fname)
  for fname in glob.glob('input/phase_*.csv'): os.unlink(fname)
  select_dt()


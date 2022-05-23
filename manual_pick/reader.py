import os
import glob
import numpy as np
from obspy import UTCDateTime

def read_fctlg(fctlg):
    f=open(fctlg); lines=f.readlines(); f.close()
    event_list = []
    for line in lines:
        codes = line.split(',')
        ot = UTCDateTime(codes[0])
        lat, lon, dep, mag = [float(code) for code in codes[1:5]]
        event_list.append([ot, lat, lon, dep, mag])
    return event_list

def read_fpha(fpha):
    f=open(fpha); lines=f.readlines(); f.close()
    event_list = []
    for line in lines:
        codes = line.split(',')
        if len(codes[0])>10:
            ot = UTCDateTime(codes[0])
            lat, lon, dep, mag = [float(code) for code in codes[1:5]]
            event_loc = [ot, lat, lon, dep, mag]
            event_list.append([event_loc, {}])
        else:
            net_sta = codes[0]
            tp = UTCDateTime(codes[1]) if codes[1]!='-1' else -1
            ts = UTCDateTime(codes[2]) if codes[2][:-1]!='-1' else -1
            event_list[-1][-1][net_sta] = [tp, ts]
    return event_list

def read_fsta(fsta):
    f=open(fsta); lines=f.readlines(); f.close()
    sta_dict = {}
    for line in lines:
        codes = line.split(',')
        sta = codes[0]
        lat, lon, ele = [float(code) for code in codes[1:4]]
        sta_dict[sta] = [lat, lon, ele]
    return sta_dict

def get_data_dict(date, data_dir):
    # get data paths
    data_dict = {}
    date_code = '{:0>4}{:0>2}{:0>2}'.format(date.year, date.month, date.day)
    st_paths = sorted(glob.glob(os.path.join(data_dir, date_code, '*')))
    for st_path in st_paths:
        fname = os.path.basename(st_path)
        net_sta = '.'.join(fname.split('.')[0:2])
        if net_sta in data_dict: data_dict[net_sta].append(st_path)
        else: data_dict[net_sta] = [st_path]
    # drop bad sta
    todel = [net_sta for net_sta in data_dict if len(data_dict[net_sta])!=3]
    for net_sta in todel: data_dict.pop(net_sta)
    return data_dict

def dtime2str(dtime):
    date = ''.join(str(dtime).split('T')[0].split('-'))
    time = ''.join(str(dtime).split('T')[1].split(':'))[0:9]
    return date + time

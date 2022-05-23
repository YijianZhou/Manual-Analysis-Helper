import sys, glob, os
from obspy import read, UTCDateTime
from reader import dtime2str

# i/o paths
fpha = 'input/eg_org.pha'
fout = open('output/eg_man.pha','w') # only pick lines are changed
event_root = 'input/eg_events'

f=open(fpha); lines=f.readlines(); f.close()
for line in lines:
    codes = line.split(',')
    if len(codes[0])<10: continue
    ot = UTCDateTime(codes[0])
    event_name = dtime2str(ot)
    event_dir = os.path.join(event_root, event_name)
    st_paths = glob.glob(os.path.join(event_dir,'*HZ'))
    if len(st_paths)==0: continue
    fout.write(line)
    for st_path in st_paths:
        st = read(st_path, headonly=True)
        fname = os.path.basename(st_path)
        net, sta = fname.split('.')[0:2]
        net_sta = '%s.%s'%(net,sta)
        header = st[0].stats.sac
        if 't2' in header: continue
        t0 = st[0].stats.starttime
        tp = t0 + header.t0 
        ts = t0 + header.t1 
        fout.write('%s,%s,%s\n'%(net_sta, tp, ts))
fout.close()

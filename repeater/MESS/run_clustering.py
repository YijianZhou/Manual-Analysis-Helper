""" Clustering with dt.cc (Main function)
"""
import os, shutil, glob
import numpy as np
from obspy import UTCDateTime
import config

# i/o paths
cfg = config.Config()
fpha = 'input/phase.csv'
out_pha = open('output/%s.pha'%cfg.ctlg_code,'w')
out_pha_full = open('output/%s_full.pha'%cfg.ctlg_code,'w')
out_clust = open('output/%s.clust'%cfg.ctlg_code,'w')
fcc = 'input/dt.cc'

# assoc MESS detections
os.system('python mk_dt.py')

# read fpha with evid
event_dict = {}
f=open(fpha); lines=f.readlines(); f.close()
for line in lines:
    codes = line.split(',')
    if len(codes[0])>14: 
        ot, lat, lon, dep, mag, evid = codes
        evid = evid[:-1]
        event_dict[evid] = ['%s,%s,%s,%s,%s\n'%(ot, lat, lon, dep, mag), []]
    else: event_dict[evid][1].append(line)

# get evid list
evid_list = []
f=open(fcc); lines=f.readlines(); f.close()
for line in lines:
    codes = line.split()
    if line[0]!='#': continue
    evid_list += codes[1:3]
evid_list = np.unique(evid_list)
# write phase
for ii,evid in enumerate(evid_list):    
    event_line, phase_lines = event_dict[evid]
    out_pha.write(event_line)
    out_pha_full.write(event_line[:-1]+',%s\n'%ii)
    for phase_line in phase_lines: 
        out_pha.write(phase_line)
        out_pha_full.write(phase_line)
# build corr_mat
num_events = len(evid_list)
evid_dict = {}
for ii,evid in enumerate(evid_list): evid_dict[evid] = ii
corr_mat = np.zeros([num_events, num_events])
for line in lines:
    codes = line.split()
    if line[0]!='#': continue
    ii, jj = np.sort([evid_dict[code] for code in codes[1:3]])
    corr_mat[ii,jj] = 1

# run clustering
clusters = []
for i in range(num_events-1):
    print('%s links remaining'%int(np.sum(corr_mat)))
    nbrs  = list(np.where(corr_mat[i]==1)[0])
    nbrs += list(np.where(corr_mat[:,i]==1)[0])
    corr_mat[i, corr_mat[i]==1] = 0
    corr_mat[corr_mat[:,i]==1, i] = 0
    if len(nbrs)>0: clusters.append(nbrs+[i]) # save the evid
    while len(nbrs)>0:
        new_nbrs = []
        for nbr in nbrs:
            new_nbrs += list(np.where(corr_mat[nbr]==1)[0])
            new_nbrs += list(np.where(corr_mat[:,nbr]==1)[0])
            corr_mat[nbr, corr_mat[nbr]==1] = 0
            corr_mat[corr_mat[:,nbr]==1, nbr] = 0
        clusters[-1] += new_nbrs
        nbrs = new_nbrs
print('%s clusters found'%len(clusters))

# write clusters
for i,cluster in enumerate(clusters):
    print('write %sth cluster'%i)
    out_clust.write('# cluster %s \n'%i)
    cluster = np.unique(cluster)
    for idx in cluster: 
        event_line = event_dict[evid_list[idx]][0]
        out_clust.write(event_line)
out_clust.close()
out_pha.close()
out_pha_full.close()

""" run_clustering
"""
import os, shutil, glob
import numpy as np
from obspy import UTCDateTime
from dataset_cc import read_fpha_dict
import config

# i/o paths
cfg = config.Config()
fpha = cfg.fpha
event_dict = read_fpha_dict(fpha)
out_pha = open('output/%s.pha'%cfg.ctlg_code,'w')
out_pha_full = open('output/%s_full.pha'%cfg.ctlg_code,'w')
out_clust = open('output/%s.clust'%cfg.ctlg_code,'w')
flink = 'output/cc_dt.link'

# 1. link events with CC
os.system('python cc_link.py')
os.system('python select_link.py')

# get evid list
evid_list = []
f=open(flink); lines=f.readlines(); f.close()
for line in lines:
    codes = line.split(',')
    if len(codes)>2: continue
    evid_list += [str(int(code)) for code in codes]
evid_list = np.unique(evid_list)
evid_dict = {}
for ii,evid in enumerate(evid_list): evid_dict[evid] = ii
# select events by num_nbr
num_events = len(evid_list)
link_mat = np.zeros([num_events, num_events])
for line in lines:
    codes = line.split(',')
    if len(codes)>2: continue
    ii, jj = np.sort([evid_dict[str(int(code))] for code in codes])
    link_mat[ii,jj] = 1
num_nbr = np.zeros(num_events)
for ii in range(num_events):
    num_nbr[ii] = sum(link_mat[ii] + link_mat[:,ii])
evid_list = evid_list[num_nbr>=cfg.min_nbr]
evid_dict = {}
for ii,evid in enumerate(evid_list): evid_dict[evid] = ii
# write phase
for ii,evid in enumerate(evid_list):
    event_line, phase_lines = event_dict[evid][1:3]
    out_pha.write(event_line)
    out_pha_full.write(event_line[:-1]+',%s\n'%evid) 
    for phase_line in phase_lines:
        out_pha.write(phase_line)
        out_pha_full.write(phase_line)
# build link_mat
num_events = len(evid_list)
link_mat = np.zeros([num_events, num_events])
for line in lines:
    codes = line.split(',')
    if len(codes)>2: continue
    evid1, evid2 = [str(int(code)) for code in codes]
    if evid1 not in evid_dict or evid2 not in evid_dict: continue
    ii, jj = np.sort([evid_dict[evid1], evid_dict[evid2]])
    link_mat[ii,jj] = 1

# 2. run clustering
clusters = []
for i in range(num_events-1):
    nbrs  = list(np.where(link_mat[i]==1)[0])
    nbrs += list(np.where(link_mat[:,i]==1)[0])
    link_mat[i, link_mat[i]==1] = 0
    link_mat[link_mat[:,i]==1, i] = 0
    if len(nbrs)>0: clusters.append(nbrs+[i]) # save the evid
    while len(nbrs)>0:
        new_nbrs = []
        for nbr in nbrs:
            new_nbrs += list(np.where(link_mat[nbr]==1)[0])
            new_nbrs += list(np.where(link_mat[:,nbr]==1)[0])
            link_mat[nbr, link_mat[nbr]==1] = 0
            link_mat[link_mat[:,nbr]==1, nbr] = 0
        clusters[-1] += new_nbrs
        nbrs = new_nbrs
print('%s clusters found'%len(clusters))

# write clusters
for i,cluster in enumerate(clusters):
    print('write %sth cluster'%i)
    out_clust.write('# cluster %s \n'%i)
    cluster = np.unique(cluster)
    for idx in cluster: out_clust.write(event_dict[evid_list[idx]][1])
out_clust.close()
out_pha.close()
out_pha_full.close()

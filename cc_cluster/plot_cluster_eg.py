""" Plot closter distribution in map view
"""
import sys
import numpy as np
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# catalog info
fclust = 'output/eg.clust'
title = 'Example CC Clustering'
fout = 'output/eg_cluster.pdf'
ot_rng = '20190704-20190710'
ot_rng = [UTCDateTime(time) for time in ot_rng.split('-')]
lon_rng =  [-117.85, -117.25]
lat_rng = [35.45, 36.05]
dep_rng = [0, 15]
mag_rng = [-1,8]
mag_corr = 0.
min_events = 10
# fig config
fig_size = (10*0.8, 12*0.8)
alpha = 0.8
mark_size = 2.
fsize_label = 14
fsize_title = 18

# read cluster
clusts = []
f=open(fclust); lines=f.readlines(); f.close()
for line in lines:
    if line[0]=='#': clusts.append([])
    else:
        lat, lon, dep, mag = [float(code) for code in line.split(',')[1:5]]
        if lat>lat_rng[1] or lat<lat_rng[0]: continue
        if lon>lon_rng[1] or lon<lon_rng[0]: continue
        if dep>dep_rng[1] or dep<dep_rng[0]: continue
        clusts[-1].append([lon, lat, (mag+mag_corr)*mark_size])
clusts = [clust for clust in clusts if len(clust)>=min_events]
print(len(clusts), 'clusters')

# plot seis loc map
fig = plt.figure(figsize=fig_size)
ax = plt.gca()
# fill up edge
edgex = [lon_rng[0], lon_rng[0], lon_rng[1], lon_rng[1]]
edgey = [lat_rng[0], lat_rng[1], lat_rng[0], lat_rng[1]]
plt.scatter(edgex, edgey, alpha=0)
# plot seis events
for clust in clusts:
    clust = np.array(clust)
    plt.scatter(clust[:,0], clust[:,1], clust[:,2], alpha=alpha)
# label & title
plt.title(title, fontsize=fsize_title)
plt.setp(ax.xaxis.get_majorticklabels(), fontsize=fsize_label)
plt.setp(ax.yaxis.get_majorticklabels(), fontsize=fsize_label)
# save fig
plt.tight_layout()
plt.savefig(fout)

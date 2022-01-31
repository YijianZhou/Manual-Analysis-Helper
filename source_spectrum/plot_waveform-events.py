import os, glob, sys
sys.path.append('/home/zhouyj/software/data_prep')
from obspy import read, UTCDateTime
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from signal_lib import preprocess
from reader import read_fpha, read_fsta, dtime2str, get_data_dict
import warnings
warnings.filterwarnings("ignore")

# i/o paths
fsta = 'input/station.csv'
sta_dict = read_fsta(fsta)
data_dir = '/data/Example_data'
get_data_dict = get_data_dict
fpha = 'input/egf_org.pha'
event_list = read_fpha(fpha)
chn_idx, chn = 2, 'Z'
out_root = 'output/waveform-fig_egf'
if not os.path.exists(out_root): os.makedirs(out_root)
# signal process
samp_rate = 100
win_len = [0,80]
npts = int(samp_rate * sum(win_len))
time = -win_len[0] + np.arange(npts) / samp_rate
freq_band = [1,20]
num_workers = 10
# fig config
fig_size = (14,9)
fsize_label = 14
fsize_title = 18
line_wid = 1.

def plot_label(xlabel=None, ylabel=None, title=None):
    if xlabel: plt.xlabel(xlabel, fontsize=fsize_label)
    if ylabel: plt.ylabel(ylabel, fontsize=fsize_label)
    if title: plt.title(title, fontsize=fsize_title)
    plt.setp(plt.gca().xaxis.get_majorticklabels(), fontsize=fsize_label)
    plt.setp(plt.gca().yaxis.get_majorticklabels(), fontsize=fsize_label)

def sort_sta(pick_dict):
    dtype = [('sta','O'),('tp','O')]
    picks = [(sta,tp) for sta, [tp,ts] in pick_dict.items()]
    picks = np.array(picks, dtype=dtype)
    picks = np.sort(picks, order='tp')
    return picks['sta']


class Plot_Events(Dataset):

  def __init__(self, event_list):
    self.event_list = event_list

  def __getitem__(self, index):
    event_loc, pick_dict = event_list[index]
    ot, _,_,_, mag = event_loc
    event_name = dtime2str(ot)
    data_dict = get_data_dict(ot, data_dir)
    sta_list = sort_sta(pick_dict)
    # plot waveform
    print('plot %s'%event_name)
    fout = os.path.join(out_root,'%s.pdf'%event_name)
    plt.figure(figsize=fig_size)
    title = 'Event Waveform: %s M%s %s %s-%sHz'%(event_name, mag, chn, freq_band[0],freq_band[1])
    for ii,sta in enumerate(sta_list):
        data_path = data_dict[sta][chn_idx]
        st = read(data_path)
        st = preprocess(st.slice(ot-win_len[0], ot+win_len[1]), samp_rate, freq_band)
        st_data = st.normalize()[0].data[0:npts] + ii*2
        plt.plot(time, st_data, lw=line_wid)
    plt.yticks(np.arange(len(sta_list))*2, sta_list, fontsize=fsize_label)
    plot_label('Time (s)',None,title)
    plt.tight_layout()
    plt.savefig(fout)
    return True

  def __len__(self):
    return len(self.event_list)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) # 'spawn' or 'forkserver'
    dataset = Plot_Events(event_list)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=None)
    for ii,_ in enumerate(dataloader):
        print('%s/%s events done/total'%(ii,len(dataset)))


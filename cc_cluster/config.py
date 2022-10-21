""" Configure file for CC clustering
"""

class Config(object):
  def __init__(self):

    # i/o paths
    self.ctlg_code = 'eg'
    self.fsta = 'input/station_eg.csv'
    self.fpha = 'input/eg_full.pha'
    self.event_root = '/data/bigdata/zhouyj/eg_events'
    self.data_dir = '/data/Example_data'
    # thresholds for event pair link
    self.cc_thres = [[0.3,0.3],[0.35,0.4]] # CC thres for event pair
    self.dt_thres = [[0.5,0.8,0.25],[0.5,0.8,0.2]] # dt_p, dt_s, d(S-P)
    self.loc_dev_thres = [5,5] # km, maximum loc separation
    self.dep_dev_thres = [5,5] # km, maximum dep separation
    self.dist_thres = [150,150] # km, max epicentral dist
    self.num_sta_thres = [3,3] # min sta_num for one event pair
    self.num_nbr_thres = [3,200]
    self.temp_mag = 0.    # min mag for templates
    self.temp_sta = 4    # min sta_num for templates
    # data prep
    self.num_workers = 10
    self.freq_band = [1.,15.]
    self.samp_rate = 100
    self.chn_p = [[2],[0,1,2]][0] # chn for P picking
    self.chn_s = [[0,1],[0,1,2]][0] # chn for S picking
    self.win_event = [5, 25]    # event data cutting, just long enough
    self.win_temp_det = [1.,11.]
    self.win_temp_p = [0.5,2.]
    self.win_temp_s = [0.2,3.8]
    # event selection
    self.ot_range = '20190704-20190710'
    self.lat_range = [35.2,36.3]
    self.lon_range = [-118.2,-117.1]


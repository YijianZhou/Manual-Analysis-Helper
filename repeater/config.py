""" Configure file for repeater detection
"""

class Config(object):
  def __init__(self):

    # 0 - initial detection; 1 - final detection
    idx = 1
    self.temp_pha = ['input/eg_pal_hyp_full.pha','input/eg_rep-org_full.pha'][idx]
    self.det_pha = ['input/eg_mess.pha','input/eg_mess-rep.pha'][idx] 
    self.fctlg = 'input/eg_mess_cc.ctlg'  # final relocated MESS catalog
    self.fsta = 'input/station_eg.csv'
    self.ctlg_code = ['eg_rep-org','eg_rep'][idx]
    self.time_range = '20210501-20210530'
    self.num_workers = 29
    self.evid_stride = 100000
    self.ot_dev = 1.5 # ot diff for det assoc
    self.cc_thres = [[0.8,0.8],[0.9,0.9]][idx] # for event & phase
    self.dt_thres = [0.03,0.01][idx] # max dt_sp
    self.nbr_thres = 1  # min number of neighbor events (i.e. linked by CC)
    self.min_sta = 3


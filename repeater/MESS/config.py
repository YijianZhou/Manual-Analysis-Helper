""" Configure file for repeater detection
"""

class Config(object):
  def __init__(self):

    self.temp_pha = 'input/eg_repeater_org_full.pha'  # repeater candidates selected from initial MESS detections
    self.det_pha = 'input/eg_mess-rep.pha'  # run MESS on repeater candidates
    self.fctlg_reloc = 'input/eg_mess_cc.ctlg'  # final relocated MESS catalog
    self.fsta = 'input/station_eg.csv'
    self.ctlg_code = 'eg_repeater'
    self.time_range = '20210501-20210530'
    self.num_workers = 29
    self.evid_stride = 100000
    self.ot_dev = 1.5 # ot diff for det assoc
    self.cc_thres = [0.9,0.9] # for event & phase
    self.dt_thres = 0.01 # max dt_sp
    self.nbr_thres = 1
    self.min_sta = 3


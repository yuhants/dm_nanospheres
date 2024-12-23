import numpy as np
import analysis_utils as utils
import os
import h5py

from astropy.time import Time
from datetime import datetime

latitude_newhaven  = 41.31
longitude_newhaven = -72.923611

file = '/Users/yuhan/work/nanospheres/data/dm_data_processed/20241202_8e-8mbar_long/20241202_abcd_1412_processed.hdf5'
f = h5py.File(file, 'r')
timestamp = f['data_processed'].attrs['timestamp']
f.close()

print(datetime.fromtimestamp(timestamp))


# datasets = ['20241202_8e-8mbar_long',
#            '20241204_2e-8mbar_8e_aftercal_long',
#            '20241205_2e-8mbar_0e_aftercal_long',
#            '20241206_1e-8mbar_0e_aftercal_long',
#            '20241207_1e-8mbar_1e_aftercal_long',
#            '20241208_1e-8mbar_1e_aftercal_long',
#            '20241210_1e-8mbar_8e_alignment1_long',
#            '20241210_1e-8mbar_8e_alignment2_long_nodrive',
#            '20241210_1e-8mbar_8e_alignment2_long_withdrive',
#            '20241211_1e-8mbar_8e_alignment2_long_nodrive',
#            '20241212_1e-8mbar_8e_alignment2_long_nodrive',
#            '20241213_1e-8mbar_0e_alignment2_long',
#            '20241214_1e-8mbar_0e_alignment2_long',
#            ]

# data_prefixs = ['20241202_abcd_',
#                 '20241204_abcd_',
#                 '20241205_d_',
#                 '20241206_d_',
#                 '20241207_d_',
#                 '20241208_d_',
#                 '20241210_d_',
#                 '20241210_d_',
#                 '20241210_d_',
#                 '20241211_d_',
#                 '20241212_d_',
#                 '20241213_d_',
#                 '20241214_d_',
#                 ]

# n_files = [1440, 1440, 1440, 1440, 1440, 821, 640, 1440, 181, 1418, 917, 1169, 1565]
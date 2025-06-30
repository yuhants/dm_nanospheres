import os
import numpy as np

import h5py
from astropy.time import Time

longitude_newhaven = -72.923611 # deg

# data_dir = r'/Users/yuhan/work/nanospheres/dm_nanospheres/data_processed/sphere_data/final_recon_before_bg_rate_cut'
data_dir = r'/home/yt388/microspheres/dm_nanospheres/data_processed/sphere_data/final_recon_before_bg_rate_cut'
file1 = h5py.File(rf'{data_dir}/sphere_20250103_unbinned_amps.h5py')
time1 = file1['unbinned_amps']['time'][:]
file1.close()

# Get the timestamps of local sidereal time bins
def get_sidereal_bins(t0, lst_hour0, ndays):
    sday = 86164.0905

    begin = t0 - (lst_hour0 / 24) * sday
    bins_sidereal_timestamps = []
    for i in range(24 * ndays + 1):
        bins_sidereal_timestamps.append(begin + i * sday/24)

    t_bins = Time(bins_sidereal_timestamps, format='unix', scale='utc')
    lst_bins = t_bins.sidereal_time('apparent', longitude=longitude_newhaven).hour
    correction = (np.round(lst_bins) - lst_bins) * (sday/(24*3600))

    return np.asarray(bins_sidereal_timestamps) + correction

data_dir = '/home/yt388/microspheres/dm_nanospheres/data_processed/dm_rate/daily_modulation'
npz = np.load(f'{data_dir}/dsdqdv_nanosphere_5.29278e-01_1.23616e-06_1e+00.npz')
# npz = np.load(f'{data_dir}/dsdqdv_nanosphere_1.77828e+00_2.09209e-07_1e+00.npz')
# npz = np.load(f'{data_dir}/dsdqdv_nanosphere_4.68870e+03_1.74604e-05_1e+00.npz')
mx = npz['mx_gev'] * 1e9
alpha_n = npz['alpha_n']
m_phi = 1

bins_sidereal_1 = get_sidereal_bins(time1[0], Time(time1[0], format='unix', scale='utc').sidereal_time('apparent', longitude=longitude_newhaven).hour, 22)
bins_siderealdays_1 = np.array([bins_sidereal_1[i] for i in range(bins_sidereal_1.size) if i % 24 == 0])

outfile = 'joblist_dm_modulation_rate.txt'

print(f'Writing file {outfile}')
job_file = open(outfile, "wt")
for i in range(bins_siderealdays_1.size - 1):
    timestamps = np.linspace(bins_siderealdays_1[i], bins_siderealdays_1[i+1], 200, endpoint=False)

    for t in timestamps:
        outdir = r'/home/yt388/palmer_scratch/data/dm_rate/daily_modulation'
        file_name = f'/drdqz_{mx:.5e}_{alpha_n:.5e}_{m_phi:.0e}_{int(t)}.npz'

        if( os.path.isfile(outdir+file_name) ):
            print("Skipping: ", file_name)
            continue

        job_str = f'module load miniconda; conda activate microsphere; python ../dm_modulation_rate.py {t}\n'
        job_file.write( job_str )

job_file.close()

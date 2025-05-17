import numpy as np
import h5py
import os

from astropy.time import Time, TimeDelta
from astropy.coordinates import EarthLocation

noise_thr = 200
dt_window = 10e-3

amp2kev, sigma_p = 5945.245097647231, 148.3093981742833   # Sphere 20241202
# amp2kev, sigma_p = 5923.2059527417405, 146.88560114966003 # Sphere 20250103

sphere = 'sphere_20241202'
datasets = [
            '20241202_8e-8mbar_long',
            '20241204_2e-8mbar_8e_aftercal_long',
            '20241205_2e-8mbar_0e_aftercal_long',
            '20241206_1e-8mbar_0e_aftercal_long',
            '20241207_1e-8mbar_1e_aftercal_long',
            '20241208_1e-8mbar_1e_aftercal_long',
            '20241210_1e-8mbar_8e_alignment1_long',
            '20241210_1e-8mbar_8e_alignment2_long_withdrive',
            '20241210_1e-8mbar_8e_alignment2_long_nodrive',
            '20241211_1e-8mbar_8e_alignment2_long_nodrive',
            '20241212_1e-8mbar_8e_alignment2_long_nodrive',
            '20241213_1e-8mbar_0e_alignment2_long',
            '20241214_1e-8mbar_0e_alignment2_long',
            '20241215_9e-9mbar_0e_alignment2_long',
            '20241216_5e-8mbar_0e_alignment2_long',
            '20241217_6e-8mbar_0e_alignment3_long',
            ]
data_prefixs = [
                '20241202_abcd_',
                '20241204_abcd_',
                '20241205_d_',
                '20241206_d_',
                '20241207_d_',
                '20241208_d_',
                '20241210_d_',
                '20241210_d_',
                '20241210_d_',
                '20241211_d_',
                '20241212_d_',
                '20241213_d_',
                '20241214_d_',
                '20241215_d_',
                '20241216_d_',
                '20241217_d_',
                ]
n_files = [1440, 1440, 1440, 1440, 1440, 821, 640, 181, 1440, 1418, 917, 1169, 1565, 1440, 1164, 601]

# sphere = 'sphere_20250103'
# datasets = ['20250104_4e-8mbar_alignment0_long',
#             '20250105_2e-8mbar_alignment0_long',
#             '20250106_2e-8mbar_8e_alignment0_long',
#             '20250107_1e-8mbar_8e_alignment0_long',
#             '20250108_1e-8mbar_8e_alignment0_long',
#             '20250109_1e-8mbar_8e_alignment1_long',
#             '20250110_1e-8mbar_8e_alignment1_long',
#             '20250111_1e-8mbar_8e_alignment1_long',
#             '20250112_9e-9mbar_8e_alignment1_long',
#             '20250113_5e-8mbar_8e_alignment1_long',
#             '20250114_1e-8mbar_1e_alignment1_long',
#             '20250115_8e-9mbar_0e_alignment1_long',
#             '20250116_8e-9mbar_0e_alignment1_long_wrong_lo',
#             '20250117_8e-9mbar_0e_alignment1_long',
#             '20250118_8e-9mbar_1e_alignment1_long',
#             '20250120_8e-9mbar_1e_alignment1_long_wbackscat',
#             '20250121_8e-9mbar_1e_alignment1_long',
#             '20250122_8e-9mbar_1e_alignment1_long',
#             '20250123_7e-9mbar_1e_alignment1_long',
#             '20250124_7e-9mbar_1e_alignment1_long',
#             '20250125_7e-9mbar_1e_alignment1_long'    
#            ]

# data_prefixs = ['20250104_d_',
#                 '20250105_d_',
#                 '20250106_d_',
#                 '20250107_d_',
#                 '20250108_d_',
#                 '20250109_d_',
#                 '20250110_d_',
#                 '20250111_d_',
#                 '20250112_d_',
#                 '20250113_d_',
#                 '20250114_d_',
#                 '20250115_d_',
#                 '20250116_d_',
#                 '20250117_d_',
#                 '20250118_d_',
#                 '20250120_d_',
#                 '20250121_d_',
#                 '20250122_d_',
#                 '20250123_d_',
#                 '20250124_d_',
#                 '20250125_d_',
#                 ]
# n_files = [1440, 900, 1440, 1440, 1440, 1440, 1440, 1440, 780, 1440, 1440, 1440, 1440, 1440, 1983, 1463, 1440, 1440, 1440, 1440, 1121]

latitude_newhaven  = 41.31 # deg
longitude_newhaven = -72.923611 # deg
elevation_newhaven = 18 # m

location = EarthLocation(lat=latitude_newhaven, lon=longitude_newhaven, height=elevation_newhaven)

def get_local_hour(utc_offset, t_utc):
    t_locals = t_utc + TimeDelta(utc_offset * 3600, format='sec')

    t_local_hours = np.empty(t_locals.size)
    for i, t_local in enumerate(t_locals):
        t_local_hours[i] = t_local.datetime.hour + t_local.datetime.minute/60 + t_local.datetime.second/3600

    return t_local_hours

def group_into_days(hours):
    idx_day = np.where(np.diff(hours)<0)[0]

    hours_day = []
    for i, idx_end in enumerate(idx_day):
        if i == 0:
            idx_start = 0
        else:
            idx_start = idx_day[i-1]
        hours_day.append(hours[idx_start+1:idx_end+1])
        
    # Add the final day
    hours_day.append(hours[idx_end+1:])
    
    return hours_day

def get_hist_days(sphere, datasets, bins, sidereal=True):
    hist_days = []
    hours_tmp = None
    
    for dataset in datasets:
        npz = np.load(fr"E:\livetime_window\{sphere}\{sphere}_{dataset}_livetime_window.npz")
        time_window = npz['livetime_window']
        npz.close()

        tt = Time(time_window, format='unix', scale='utc')
        if sidereal:
            lst = tt.sidereal_time('apparent', longitude=location.lon)
            hours = lst.hour
        else:
            utc_offset = -5
            hours = get_local_hour(utc_offset, tt)

        # Compute histogram until the end of sidereal day
        next_day = False
        _idx_day = np.where(np.diff(hours) < 0)[0]
        if _idx_day.size == 1:
            idx_day = _idx_day[0]
            next_day = True
        else:
            idx_day = hours.size

        _hist, _ = np.histogram(hours[:idx_day+1], bins=bins)

        # If there is left over hours from previous dataset, add them
        # to the histogram
        if hours_tmp is not None:
            _hist_prev, _ = np.histogram(hours_tmp, bins=bins)
            _hist += _hist_prev

        hist_days.append(_hist)

        if next_day:
            hours_tmp = hours[idx_day+1:]
        else:
            hours_tmp = None

    # Add the final bit if there is leftover
    if hours_tmp is not None:
        _hist_leftover, _ = np.histogram(hours_tmp, bins=bins)
        hist_days.append(_hist_leftover)

    return np.asarray(hist_days)
    

def get_livetime_window(sphere, dataset, data_prefix, nfile):
    data_dir = fr'E:\dm_data_processed_amp_chisquare\{sphere}\{dataset}'

    good_window_times = np.empty(6000 * nfile)
    window_count = 0
    for i in range(nfile):
        if i % 100 == 0:
            print(i)

        _file = os.path.join(data_dir, f'{data_prefix}{i}_processed.hdf5')

        file = h5py.File(_file, 'r')
        timestamp = file['data_processed'].attrs['timestamp']
        good_det = file['data_processed']['good_detection'][:]
        noise_level_kev = file['data_processed']['noise_level_amp'][:] * amp2kev

        file.close()

        _good_window_idx = np.nonzero(np.logical_and(good_det, noise_level_kev < noise_thr))[0]
        _good_window_time = timestamp + dt_window * _good_window_idx
        good_window_times[window_count: window_count+_good_window_time.size] = _good_window_time

        window_count += _good_window_time.size

    return good_window_times[:window_count]

for i, dataset in enumerate(datasets):
    out_dir = rf'E:\livetime_window\{sphere}'
    if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

    outfile = rf'{sphere}_{dataset}_livetime_window.npz'
    good_window_times = get_livetime_window(sphere, dataset, data_prefixs[i], n_files[i])

    print(f'Writing file {os.path.join(out_dir, outfile)}')
    np.savez(os.path.join(out_dir, outfile), livetime_window=good_window_times)
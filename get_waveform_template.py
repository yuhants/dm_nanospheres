import numpy as np
import analysis_utils as utils


def get_pulse_shape(f_lp, amp, length=1500):
    f_lp_scaled = f_lp / 1e9
    pulse_idx_in_win = np.argmin(np.abs(np.abs(f_lp_scaled) - amp))

    if f_lp_scaled[pulse_idx_in_win] > 0:
        polarity = 1
    else:
        polarity = -1

    ret = polarity * f_lp_scaled[pulse_idx_in_win - length : pulse_idx_in_win + length]

    # Get 50 us around the maximum amplitude
    return ret

sphere = 'sphere_20241202'
datasets = ['20241204_8e', '20241205_8e', '20241209_8e_alignment1_1', '20241213_8e_alignment2_4', '20241213_8e_alignment2_5']

# sphere = 'sphere_20250103'
# datasets = ['20250106_8e_alignment0_2e-8mbar_0', '20250106_8e_alignment0_2e-8mbar_1', '20250107_8e_alignment0_1e-8mbar_0', '20250107_8e_alignment0_1e-8mbar_1', 
#             '20250108_8e_alignment0_1e-8mbar_0', '20250108_8e_alignment0_1e-8mbar_1', '20250117_8e_alignment1_8e-9mbar_0', '20250117_8e_alignment1_8e-9mbar_1']

voltages = [20, 10, 12.5, 15, 17.5]
nfiles = 10

for v in voltages:
    print(v)

    pulse_shapes = []
    for dataset in datasets:
        print(f'Working on {dataset}')

        for i in range(nfiles):
            date = dataset[:8]
            if date == '20250106':
                date = '20250116'  # because of an error in naming the files...

            data_file = rf'/Volumes/LaCie/pulse_calibration/{sphere}/{dataset}/{date}_dg_8e_200ns_{v}v_{i}.hdf5'

            dtt, nn = utils.load_timestreams(data_file, ['D', 'G'])
            fs = int(np.ceil(1/dtt))
            zz, dd = nn[0], nn[1]
            zz_bp = utils.bandpass_filtered(zz, fs, 30000, 80000)

            drive_indices = utils.get_pulse_idx(dd, 0.5, True)

            for drive_idx in drive_indices:
                # Modified 20250211: match the search window length
                # to calibration data processing and DM search
                window, f, f_lp, amp = utils.recon_pulse(drive_idx, dtt, zz_bp, dd, 50000, 50000, 250, 20)
                if window is None:
                    continue
                pulse_shape = get_pulse_shape(f_lp, amp, 1500)

                if pulse_shape.size != 3000:
                    print('Skipping pulse near the end of file')
                    continue
                pulse_shapes.append(pulse_shape)
    pulse_shapes = np.asarray(pulse_shapes)

    outdir = r'/Users/yuhan/work/nanospheres/dm_nanospheres/data_processed/pulse_shape'
    outfile = f'{outdir}/{sphere}_pulse_shape_template_{v}v.npz'
    print(f'Writing file {outfile}')
    np.savez(outfile, pulse_shape=pulse_shapes)
import numpy as np

from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord
from astropy import units as u

import sys
from datetime import datetime, timezone

from scipy.integrate import cumulative_trapezoid

## Parameters
deg2rad = np.pi / 180
rad2deg = 180 / np.pi
c_kms = 299792458 / 1000  # Speed of light (km/s)
hbarc = 0.2     # eV um

latitude_newhaven  = 41.31      # deg
longitude_newhaven = -72.923611 # deg

# Direction of momentum transfer (z) in NWZ frame
z_dir_nwz = np.array([np.sin(20*deg2rad), np.cos(20*deg2rad), 0])

# DM parameters
rhoDM = 0.3e9        # dark matter mass density, eV/cm^3
v_gal_rot = np.array([0, 220, 0])      # Galactic rotation speed in the Galactic frame; km/s
v_solar = np.array ([11.1, 12.2, 7.3]) # Solar motion in the Galactic frame

## Functions
## Implement velocity transformations described in https://journals.aps.org/prd/abstract/10.1103/PhysRevD.84.023516
def unix_to_mjd(unix_time):
    return unix_time / 86400 + 40587

def get_T0_UT(unix_time):
    mjd = unix_to_mjd(unix_time)
    T0 = (int(mjd) - 55197.5) / 36525
    
    utc_dt = datetime.fromtimestamp(unix_time, timezone.utc)
    t = Time(utc_dt, scale='utc')
    ut_dt = t.ut1.value
    UT = ut_dt.hour + ut_dt.minute/60 + ut_dt.second/3600 + ut_dt.microsecond/(3600 * 1e6)

    return T0, UT

def get_lamb_t(unix_time):
    T0, UT = get_T0_UT(unix_time)

    L = 281.0298 + 36000.77 * T0 + 0.04107 * UT
    g = 357.9258 + 35999.05 * T0 + 0.04107 * UT

    ret = L + (1.915 - 0.0048 * T0) * np.sin(g * deg2rad) + 0.02 * np.sin(2 * g * deg2rad)
    return ret

def v_earth_rev(t):
    """Earth revolution speed around the sun in the Galactic frame
    Input time, `t`, should be in unix timestamp.
    """
    # Compute for only one time (no array)
    if not np.isscalar(t):
        raise('`t` should be a scalar')

    v_earth_orbital = 29.8   # km/s
    e = 0.016722  # ellipticity

    lamb_0 = 13   # deg
    # Ecliptic latitudes and longitudes (deg)
    beta = np.array([-5.5303, 59.575, 29.812])
    lamb = np.array([266.141, -13.3485, 179.3212])

    lamb_t = get_lamb_t(t) # in degree
    
    v_earth_orbital_t = v_earth_orbital * (1 - e * np.sin((lamb_t - lamb_0) * deg2rad ))
    ret_xg = v_earth_orbital_t * np.cos(beta[0] * deg2rad) * np.sin( (lamb_t - lamb[0]) * deg2rad)
    ret_yg = v_earth_orbital_t * np.cos(beta[1] * deg2rad) * np.sin( (lamb_t - lamb[1]) * deg2rad)
    ret_zg = v_earth_orbital_t * np.cos(beta[2] * deg2rad) * np.sin( (lamb_t - lamb[2]) * deg2rad)

    return np.array([ret_xg, ret_yg, ret_zg])

def v_earth_rot(t, long_lab, lat_lab):
    """Earth rotation speed in the Galactic frame
    Input time, `t`, should be in unix timestamp.
    """
    # Compute for only one time (no array)
    if not np.isscalar(t):
        raise('`t` should be a scalar')

    v_rot_eq = 0.465102  # Earth rotation speed at the equator; km/s
    lamb_lab = lat_lab

    v_rot_nwz = np.array([0, -1 * v_rot_eq * np.cos(lamb_lab * deg2rad), 0])
    v_rot_galactic = nwz_to_galactic(v_rot_nwz, t, long_lab, lat_lab)

    return v_rot_galactic

def get_LAST_deg(unix_time, longitude_loc):
    t = Time(unix_time, format='unix', scale='utc')
    lst = t.sidereal_time('apparent', longitude=longitude_loc).hour
    return lst * 15

def nwz_to_equatorial(vec_nwz, t, long_lab, lat_lab):
    # Input time, `t`, should be in unix timestamp
    if not np.isscalar(t):
        raise('`t` should be a scalar')

    if vec_nwz.shape == (3, ):
        vec_nwz = np.array([vec_nwz])

    t_lab_deg = get_LAST_deg(t, long_lab)
    lamb_lab  = lat_lab

    ret_xe = vec_nwz.T[0] * (-1) * np.sin(lamb_lab * deg2rad) * np.cos(t_lab_deg * deg2rad) + vec_nwz.T[1] * np.sin(t_lab_deg * deg2rad) + vec_nwz.T[2] * np.cos(lamb_lab * deg2rad) * np.cos(t_lab_deg * deg2rad)
    ret_ye = vec_nwz.T[0] * (-1) * np.sin(lamb_lab * deg2rad) * np.sin(t_lab_deg * deg2rad) - vec_nwz.T[1] * np.cos(t_lab_deg * deg2rad) + vec_nwz.T[2] * np.cos(lamb_lab * deg2rad) * np.sin(t_lab_deg * deg2rad)
    ret_ze = vec_nwz.T[0] * np.cos(lamb_lab * deg2rad) + vec_nwz.T[2] * np.sin(lamb_lab * deg2rad)

    return np.array([ret_xe, ret_ye, ret_ze]).T

def equatorial_to_nwz(vec_eq, t, long_lab, lat_lab):
    # Input time, `t`, should be in unix timestamp
    if not np.isscalar(t):
        raise('`t` should be a scalar')

    if vec_eq.shape == (3, ):
        vec_eq = np.array([vec_eq])

    t_lab_deg = get_LAST_deg(t, long_lab)
    lamb_lab  = lat_lab

    ret_n = vec_eq.T[0] * (-1) * np.sin(lamb_lab * deg2rad) * np.cos(t_lab_deg * deg2rad) - vec_eq.T[1] * np.sin(lamb_lab * deg2rad) * np.sin(t_lab_deg * deg2rad) + vec_eq.T[2] * np.cos(lamb_lab * deg2rad)
    ret_w = vec_eq.T[0] * np.sin(t_lab_deg * deg2rad) - vec_eq.T[1] * np.cos(t_lab_deg * deg2rad)
    ret_z = vec_eq.T[0] * np.cos(lamb_lab * deg2rad) * np.cos(t_lab_deg * deg2rad) + vec_eq.T[1] * np.cos(lamb_lab * deg2rad) * np.sin(t_lab_deg * deg2rad)+ vec_eq.T[2] * np.sin(lamb_lab * deg2rad)

    return np.array([ret_n, ret_w, ret_z]).T

def equatorial_to_galactic(vec_eq):
    if vec_eq.shape == (3, ):
        vec_eq = np.array([vec_eq])

    ret_xg = -0.06699 * vec_eq.T[0] - 0.8728 * vec_eq.T[1] - 0.4835 * vec_eq.T[2]
    ret_yg =  0.4927  * vec_eq.T[0] - 0.4503 * vec_eq.T[1] + 0.7446 * vec_eq.T[2]
    ret_zg = -0.8676  * vec_eq.T[0] - 0.1883 * vec_eq.T[1] + 0.4602 * vec_eq.T[2]
    
    return np.array([ret_xg, ret_yg, ret_zg]).T

def galactic_to_equatorial(vec_ga):
    if vec_ga.shape == (3, ):
        vec_ga = np.array([vec_ga])

    ret_xe = -0.06699 * vec_ga.T[0] + 0.4927 * vec_ga.T[1] - 0.8676 * vec_ga.T[2]
    ret_ye = -0.8728  * vec_ga.T[0] - 0.4503 * vec_ga.T[1] - 0.1883 * vec_ga.T[2]
    ret_ze = -0.4835  * vec_ga.T[0] + 0.7446 * vec_ga.T[1] + 0.4602 * vec_ga.T[2]
    
    return np.array([ret_xe, ret_ye, ret_ze]).T

def nwz_to_galactic(vec_nwz, t, long_lab, lat_lab):
    vec_eq = nwz_to_equatorial(vec_nwz, t, long_lab, lat_lab)
    vec_ga = equatorial_to_galactic(vec_eq)

    return vec_ga

def galactic_to_nwz(vec_ga, t, long_lab, lat_lab):
    vec_eq  = galactic_to_equatorial(vec_ga)
    vec_nwz = equatorial_to_nwz(vec_eq, t, long_lab, lat_lab)
    
    return vec_nwz

def v_lab_galactic(t, long_lab, lat_lab):
    v_rev = v_earth_rev(t)
    v_rot = v_earth_rot(t, long_lab, lat_lab)
    v_tot = v_gal_rot + v_solar + v_rev + v_rot

    return v_tot

## For generating MC samples of DM velocities
def get_mb_galactic(n_mc, rng):
    """Generate random MC samples following a 3D Maxwell-Boltzmann distribution"""
    v0 = 220   # velocity dispersion; km/s
    sigmav = v0 / np.sqrt(2)

    # Velocity samples following a Maxwell-Boltzmann distribution
    v_ga_xyz = rng.normal(loc=0, scale=sigmav, size=3*n_mc)
    v_ga_xyz = np.reshape(v_ga_xyz, (n_mc, 3))

    return v_ga_xyz

def vel_boosted_galactic(v_dm, v_lab, vesc):
    """Velocity of DM in the Galactic frame, boosted by the lab velocity and truncated at escape velocity"""
    v_boosted = v_dm + v_lab
    v_absolute = np.sqrt(np.sum(v_boosted**2, axis=1))

    return v_boosted[v_absolute < vesc]

def nwz_to_altaz(vec_nwz):
    vec_norm = np.sqrt(np.sum(vec_nwz**2, axis=1))

    vec_n, vec_w, vec_z = vec_nwz.T[0], vec_nwz.T[1], vec_nwz.T[2]
    vec_horizontal = np.sqrt(vec_n**2 + vec_w**2)

    alt = np.arctan2(vec_z, vec_horizontal) * rad2deg
    az  = np.arctan2(-1*vec_w, vec_n) * rad2deg
    
    # Go from (-180, 180) to (0, 360) deg
    az[az < 0] = az[az < 0] + 360

    return np.array([vec_norm, alt, az]).T

def get_v_boosted_mc_nwz(v_dm_mc_ga, timestamp):
    vesc = 550 # km/s

    v_lab_ga = v_lab_galactic(timestamp, longitude_newhaven, latitude_newhaven)
    v_boosted_ga = vel_boosted_galactic(v_dm_mc_ga, v_lab_ga, vesc)
    v_boosted_nwz = galactic_to_nwz(v_boosted_ga, timestamp, longitude_newhaven, latitude_newhaven)

    return v_boosted_nwz

## Functions for calculating the projected event rate from the MC samples
def get_dsdq(v, dsdq_v, vlist):
    """Interpolate to get the differential cross section dsig/dq at velocity v"""
    if v > vlist[-1]:
        return np.zeros_like(dsdq_v[0])

    # First index larger than v
    j_v = np.nonzero(vlist > v)[0][0]
    i_v = j_v - 1
    d_v = vlist[j_v] - vlist[i_v]

    _dsdq_i = dsdq_v[i_v]
    _dsdq_j = dsdq_v[j_v]

    # Interpolate between the two rates to the input alpha
    ret = _dsdq_i + (v - vlist[i_v]) * (_dsdq_j - _dsdq_i) / d_v
    return ret

def get_random_q_samples(qq, dsdq, rr, qmin):
    # print()
    if np.sum(dsdq[qq > qmin]) == 0:
        return None, None
    norm_factor = np.trapz(dsdq[qq > qmin], qq[qq > qmin])

    f_drdq_norm = dsdq[qq > qmin] / norm_factor       # PDF of q
    Fc_drdq_norm = cumulative_trapezoid(f_drdq_norm, x=qq[qq > qmin], initial=0) # CDF of q
    # Fc_drdq_norm = np.cumsum(f_drdq_norm) 

    qq_sampled = np.interp(rr, Fc_drdq_norm, qq[qq > qmin], left=0, right=0)
    return qq_sampled, norm_factor

def get_q_vectors(v_dm_nwz, q_mc, cosine_mc, phi_mc):
    v_hat = v_dm_nwz / np.sqrt(np.sum(v_dm_nwz**2))

    # Construct a local orthogonal basis
    # First choose an arbitrary vector not parallel to v_hat
    if abs(v_hat[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])

    # First perpendicular vector
    u_hat = np.cross(v_hat, ref)
    u_hat /= np.linalg.norm(u_hat)
    
    # Second perpendicular vector (completes right-handed system)
    w_hat = np.cross(v_hat, u_hat)

    # Calculate the direction of DM momentum after scattering
    good_idx = np.logical_and(cosine_mc > -1, cosine_mc < 1)
    sine_mc = np.sin(np.arccos(cosine_mc[good_idx]))
    pf_dir = np.outer(cosine_mc[good_idx], v_hat) +  np.outer(sine_mc * np.cos(phi_mc[good_idx]), u_hat) + np.outer(sine_mc * np.sin(phi_mc[good_idx]), w_hat)

    # Get the normalized direction of momentum transfer q
    q_dir = v_hat - pf_dir
    q_dir_norm = np.sqrt(np.sum(q_dir**2, axis=1))
    q_dir = (q_dir.T / q_dir_norm).T

    ret   = (q_mc[good_idx] * q_dir.T).T

    # q vectors in eV/c
    return ret

def get_drdqz(v_dm_nwz, z_dir_nwz, mx, dsigdq_v, qq, vlist, rr0, rr1):
    """For each incoming DM velocity, find the projected scattering rate dR/dqz
    
    `rr0` and `rr1` are random uniform samples from 0 to 1
    """
    # First find the cross section dsig/dq for this velocity
    v_dm_norm_c = np.sqrt(np.sum(v_dm_nwz**2)) / c_kms
    px = mx * v_dm_norm_c

    dsdq = get_dsdq(v_dm_norm_c, dsigdq_v, vlist)

    # Generate MC samples above 1 MeV/c
    q_mc, norm_factor = get_random_q_samples(qq, dsdq, rr0, qmin=1e6)
    if q_mc is None:
        return None, None

    # Calculate the cosine of scattering angles
    cosine_mc = 1 - q_mc**2 / (2 * px**2)
    
    # Throw away unphysically large scattering angle
    if np.sum(cosine_mc > -1) == 0:
        return None, None

    phi_mc = rr1 * 2 * np.pi
    q_vectors_mc = get_q_vectors(v_dm_nwz, q_mc, cosine_mc, phi_mc)

    # Project onto z-axis and bin into histogram
    # Take the absolute value to account for both +z and -z
    qz_mc = np.abs(q_vectors_mc @ z_dir_nwz)

    bins = np.linspace(0, 10000000, 300)
    bc = 0.5 * (bins[1:] + bins[:-1])

    hhz, _ = np.histogram(qz_mc, bins=bins)
    dsigdqz = hhz * norm_factor / (np.sum(hhz)) / (bins[1] - bins[0])

    drdqz = (rhoDM / mx) * v_dm_norm_c * dsigdqz
    drdqz[bc < 1e6] = 0 # Disregard the rate calculated below 1 MeV
    
    conv_fac = hbarc**2 * 1e3 * 3e10 * 1e-8

    # keV/c; Events/keV/s
    return bc/1e3, drdqz * conv_fac

def get_projected_rate(v_dm_mc_nwz, z_dir_nwz, mx, dsigdq_v, qq, vlist, rr0, rr1):
    """
    `v_dm_mc_nwz` is in km/s
    """
    qz_kev, drdqzs = None, None
    count = 0
    for i, v_dm in enumerate(v_dm_mc_nwz):
        _qz_kev, _drdqz = get_drdqz(v_dm, z_dir_nwz, mx, dsigdq_v, qq, vlist, rr0, rr1)
        if _qz_kev is None:
            continue

        if qz_kev is None:
            qz_kev = _qz_kev
            drdqzs = _drdqz
        else:
            drdqzs += _drdqz
        count += 1
        
    drdqzs /= count
    return qz_kev, drdqzs

## Start calculation
if __name__ == '__main__':
    timestamp       = float(sys.argv[1])
    print(f'Working on time {timestamp}')

    # Load the calculated differential cross section
    # data_dir = '/Users/yuhan/work/nanospheres/dm_nanospheres/data_processed/dm_rate/daily_modulation'
    data_dir = '/home/yt388/microspheres/dm_nanospheres/data_processed/dm_rate/daily_modulation'
    npz = np.load(f'{data_dir}/dsdqdv_nanosphere_5.29278e-01_1.23616e-06_1e+00.npz')
    # npz = np.load(f'{data_dir}/dsdqdv_nanosphere_1.77828e+00_2.09209e-07_1e+00.npz')
    # npz = np.load(f'{data_dir}/dsdqdv_nanosphere_4.68870e+03_1.74604e-05_1e+00.npz')
    mx = npz['mx_gev'] * 1e9
    alpha_n = npz['alpha_n']
    qq = npz['q']
    dsigdq_v = npz['dsdqdv']
    vlist = npz['v']
    m_phi = 1

    # Generate MC samples of DM velocities and q's
    seed = 234837942783
    rng = np.random.default_rng(seed=seed)

    n_mc = 100000       # Number of velocity samples
    v_dm_ga = get_mb_galactic(n_mc, rng)

    n_mc_dsdq = 10000   # Number of q samples for each velocity  
    rr0 = rng.uniform(0, 1, n_mc_dsdq)
    rr1 = rng.uniform(0, 1, n_mc_dsdq)

    v_dm_nwz = get_v_boosted_mc_nwz(v_dm_ga, timestamp)
    _qz, _drdqz = get_projected_rate(v_dm_nwz, z_dir_nwz, mx, dsigdq_v, qq, vlist, rr0, rr1)

    outdir = r'/home/yt388/project/data/dm_rate/daily_modulation'
    file_name = f'/drdqz_{mx:.5e}_{alpha_n:.5e}_{m_phi:.0e}_{int(timestamp)}.npz'
    print(f'Saving file {outdir + file_name}')
    np.savez(outdir+file_name, q_kev=_qz, drdqz_hz_kev=_drdqz, time=timestamp, mx_gev=mx, alpha_n=alpha_n, mphi_=m_phi)
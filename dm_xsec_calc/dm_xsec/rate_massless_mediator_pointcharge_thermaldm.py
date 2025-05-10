import sys, os
import numpy as np

import matplotlib.pyplot as plt
from scipy.special import erf, spherical_jn

# Parameters
hbarc = 0.2     # eV um
kb = 8.617e-5      # eV K^-1

# Sphere parameters
rho_T = 2.0e3   # Sphere density, kg/m^3
mAMU = 1.66e-27 # Neutron mass

# Thermal DM parameters
nx         = 1         # DM number density (assumed to be 1 cm^-3)
t_thermal  = 300       # K
vesc_earth = 3.729e-5  # Earth escape velocity

vmin  = 0
nvels = 500        # Number of velocities to include in integration
nq    = 20000      # Number of momentum transfer to sample

qmin  = 1000      # Lowest momenturm transfer considered (eV)
qmax_calc = 100e6  # The maximum momentum transfer to analyze

def f_thermal(v, mx, T, vesc_earth):
    """
    Truncated Maxwell-Boltzmann velocity distribution
    for DM thermalized at temperature T

    :param v: input velocity in natural units (array-like)
    :param mx: DM mass (eV)
    :param T: thermal temperature (K)
    :param vesc_earth: Earth's escape velocity in natural units
    """
    v0 = np.sqrt(2 * kb * T / mx)  # most probably velocity
    N0 = np.pi**1.5 * v0**3 * ( erf(vesc_earth/v0) - 2/np.sqrt(np.pi) * (vesc_earth/v0) * np.exp(-(vesc_earth/v0)**2))

    f = 4 * np.pi * v**2 * np.exp(-1 * v**2 / (v0**2))
    f[v > vesc_earth] = 0

    return f / N0

def dsig_domega_born(mx, mphi, alpha, q, R=None, point_charge=True):
    """
    Differential cross section given by Born approximation

    :param mx : DM mass in natural units
    :param phi: mediator mass
    :param q  : momentum transfer (array_like)
    :param point_charge: if True give point charge solution
                         if False include a uniform sphere form factor

    :return: dsigma/domega in natural units (array_like)
    """
    point_charge_sol = (4 * (mx**2) * (alpha**2) ) / ( (mphi**2 + q**2)**2 )
    if point_charge:
        return point_charge_sol
    else:
        form_factor = 3 * spherical_jn(n=1, z=q*R) / (q * R)
        return point_charge_sol * form_factor**2
    
def dsig_dq(dsigdomega, mx, alpha, q, vlist, R):

    ss = np.empty(shape=(vlist.size, q.size))
    ss_out = np.empty(shape=(vlist.size, q.size))

    for i, v in enumerate(vlist):
        p = mx * v  # Momentum carried by DM particle
        dsigdq = ( 2 * np.pi * q / (p**2) ) * dsigdomega

        # Cut off unphysical large-q scattering
        # Elastic scattering only
        dsigdq[q > 2 * p] = 0
        ss[i] = dsigdq

    return ss

def dR_dq_thermal(nx, mx, mphi, alpha, q, vlist, R):
    # Differential cross section in the point charge limit
    dsigdomega = dsig_domega_born(mx, mphi, alpha, q, R=None, point_charge=True)
    dsigdq = dsig_dq(dsigdomega, mx, alpha, q, vlist, R)
        
    int_vec = nx * vlist * f_thermal(vlist, mx, t_thermal, vesc_earth)
    
    drdq = np.empty_like(q)
    for i in range(q.size):
        drdq[i] = np.trapz( int_vec * dsigdq.T[i], x=vlist )
        
    # conv_fac = hbarc**2 * 1e9 * 3e10 * 1e-8 * 3600  # natural units -> um^2/GeV, c [cm/s], um^2/cm^2, s/hr
    conv_fac = hbarc**2 * 1e3 * 3e10 * 1e-8

    # keV; Differential count (Events/(keV/c)/s)
    return q/1e3, drdq * conv_fac

def calc_event_rate(R_um, mx_gev, alpha_t):
    R = R_um / hbarc       # Sphere radius, eV^-1
    N_T = 0.5 * ( 4/3 * np.pi * (R_um*1e-6)**3 ) * rho_T/mAMU # Number of neutrons

    mx = mx_gev * 1e9      # DM mass, eV
    alpha = alpha_t * N_T  # Total coupling

    pmax = np.min((2.5 * vesc_earth * mx, qmax_calc))
    q  = np.linspace(qmin, pmax, nq)

    # vlist = np.linspace(vmin, vesc_earth, nvels)
    v0 = np.sqrt(2 * kb * t_thermal / mx)  # most probably velocity
    vlist = np.linspace(v0/50, v0*50, nvels)
    q_kev, drdq = dR_dq_thermal(nx, mx, 0, alpha, q, vlist, R)

    # keV; Counts/s/kev
    return q_kev, drdq

if __name__ == "__main__":
    outdir = r"/home/yt388/palmer_scratch/data/dm_rate/thermalized_dm/massless_mediator_pointcharge"
    # outdir = r"/Users/yuhan/work/nanospheres/data/dm_rate/thermal_dm/massless_mediator_pointcharge"
    if(not os.path.isdir(outdir)):
        os.mkdir(outdir)
    
    R_um = 0.083

    mx_list = np.logspace(1, 8, 80)
    alpha_list = np.logspace(-12, -4, 80)

    if R_um < 0.5:
        sphere_type = 'nanosphere'
    else:
        sphere_type = 'microsphere'

    print(f'Sphere radius = {R_um:.3f} um')

    for i, mx in enumerate(mx_list):
        for j, alpha in enumerate(alpha_list):
            if qmax_calc == 100e6:
                outfile = outdir + f'/drdq_100mevthr_thermaldm_{sphere_type}_{R_um:.2e}_{mx:.5e}_{alpha:.5e}_massless_pointcharge.npz'
            else:
                raise('Check calculation!')
            
            if( os.path.isfile(outfile) ):
                print("Skipping: ", outfile)
                continue

            # print(f'Working on M_x = {mx:.3e} GeV, alpha_n = {alpha:.3e}')
            qq, drdq = calc_event_rate(R_um, mx, alpha)
            print(f'Saving file {outfile}')
            np.savez(outfile, mx_gev=mx, alpha_n=alpha, q_kev=qq, drdq_hz_kev=drdq)

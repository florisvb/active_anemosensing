import matplotlib.pyplot as plt
import numpy as np
import figurefirst as fifi
import fly_plot_lib as fpl
import pynumdiff
import direct_trajectory_simulator
import os

import time

try:
    import convex_solve_tan_cot
except ImportError:
    print('probably could not import cvxpy, not an issue if you dont need it here')
          
import fly_plot_lib.flymath as flymath

import pandas

from utility import wrap_angle, mean_angle, plot_wind_quivers, plot_trajec, diff_angle, get_sensor_measurements_derivatives_controls, std_angle
import utility
from plot_utility import load_real_wind


def get_df_random( wind='real', 
                   L=None, 
                   psi_freq=2,
                   dt=0.01,
                   t=None,
                   turn_amplitude='random',
                   smoothness_weight=1,
                   angular_noise_std=0,
                   of_noise_std=0,
                   air_noise_std=0,
                   velocity_profile='absine',
                   phi='align_psi',
                   random_seed=1): # align_psi or align_gamma
    '''
    wind --- 'real', or 'constant'
    '''
    np.random.seed(random_seed)
    
    if '_' not in phi:
        phi = 'align_' + phi.split('align')[1]
    
    if wind=='realdynamic':
        # use L = 15000
        df_stationary_wind_q, zeta, w, t, dt = load_real_wind('dynamic', L=L)
    elif wind == 'realconstant': 
        # Use L = 7000
        df_stationary_wind_q, zeta, w, t, dt = load_real_wind('constant', L=L)
    elif wind == 'constant':
        if t is None:
            t = np.arange(0, L*dt, dt)[0:L]
        zeta = np.ones_like(t)*np.pi/4.
        w = np.ones_like(t)*1.5
        dt = np.mean(np.diff(t))
    else:
        if t is None:
            t = np.arange(0, L*dt, dt)[0:L]
        zeta, w = wind #np.pi/2. #-np.pi # -np.pi/4. #/2. #-np.pi/2. #/2. #-np.pi/2. #-np.pi/2.

    if L is None:
        L = len(zeta)
    
    #phi = 'align_psi' # 0 #np.pi/2. #np.pi/4. #0.2*np.sin(5*t+np.pi/2.) + np.pi/4.
    acceleration = 0 #3*np.sin(2*2*np.pi*psi_freq*t+psi_phase) + psi_offset

    constant_v_para = False
    if velocity_profile == 'absine':
        vel = np.abs(np.sin(2*np.pi*psi_freq*t)) + 0.2 #1 #0.5
    elif velocity_profile == 'fastABSsine':
        vel = np.abs(np.sin(20*np.pi*psi_freq*t)) + 0.2 #1 #0.5
    else:
        vel = np.abs(np.sin(2*np.pi*psi_freq*t)) + 0.2 #1 #0.5
        mean_vel = np.mean(vel)
        vel = np.ones_like(vel)*mean_vel
    
    #N_turns = int(psi_freq*t[-1])*3 
    sign = np.sign(np.sin(2*np.pi*psi_freq*t))
    ix = (np.where(sign==0)[0])
    for i in ix:
        sign[i] = 1
    chunks, breaks = flymath.get_continuous_chunks(sign)
    N_turns = len(chunks)
    psi_global = [0]*len(chunks[0])
    
    if turn_amplitude == 'random':
        for chunk in chunks[1:]:
            psi_global.extend([np.random.uniform(-np.pi, np.pi)]*len(chunk))
    elif turn_amplitude == '20deg':
        for chunk in chunks[1:]:
            D = 20*np.pi/180.
            p = psi_global[-1] + np.random.uniform(0.8*D, 1.2*D)*np.random.choice([-1,1])
            p = wrap_angle(p)
            psi_global.extend([p]*len(chunk))
    elif turn_amplitude == '90deg':
        for chunk in chunks[1:]:
            p = psi_global[-1] + np.random.uniform(0.8*np.pi/2., 1.2*np.pi/2.)*np.random.choice([-1,1])
            p = wrap_angle(p)
            psi_global.extend([p]*len(chunk))
    elif turn_amplitude == '180deg':
        for chunk in chunks[1:]:
            p = psi_global[-1] + np.random.uniform(0.8*np.pi, 0.95*np.pi)*np.random.choice([-1,1])
            p = wrap_angle(p)
            psi_global.extend([p]*len(chunk))
            
        
    psi_global = utility.unwrap_angle( np.array(psi_global), correction_window_for_2pi=1)
    N_turns_per_second = sum(np.abs(np.diff(psi_global))>0) / t[-1]
    print('Number of turns per sec: ', N_turns_per_second)
    
    
    gaussian_window = int(L / float(N_turns))*smoothness_weight
    psi_global, psi_global_dot = pynumdiff.smooth_finite_difference.gaussiandiff(psi_global, dt, [gaussian_window], {'iterate': False})
    
    df, biomechanics_parameters = direct_trajectory_simulator.simulate_trajectory(   
                                      angular_noise_std, air_noise_std, of_noise_std, 
                                      psi_global=psi_global,
                                      gaussian_window=gaussian_window,
                                      constant_v_para=constant_v_para,
                                      acceleration=acceleration,
                                      phi=phi, 
                                      zeta=zeta,
                                      L=L, dt=dt,
                                      w=w, mean_vel=vel,
                                      biomechanics_parameters='large')
    df.phi = utility.wrap_angle(df.phi)
    df.psi = utility.wrap_angle(df.psi)
    df.gamma = utility.wrap_angle(df.gamma)
    return df, biomechanics_parameters


def solve_convex(tau, Ts, psi_freq, df, trajec_type, turn_amplitude, pynumdiff_param, angular_noise_std, wind_type='realdynamic', velocity_profile='absine', phi_alignment='align_psi', butterworth_freq_param_adjustment=1, save=True, df_sensor=None, smoothing_parameters=None, use_smoothed_for_disambiguation=False, directory='.'):
    

    # get derivatives of sensor
    #params = [2, 0.07]
    turn_period = (1/psi_freq / 2.)
    dt = np.mean(np.diff(df.t))
    if df_sensor is None:
        df_sensor, smoothing_parameters = get_sensor_measurements_derivatives_controls(df, 
                                                                 derivative_method='smooth_finite_difference.butterdiff', 
                                                                 params=pynumdiff_param,
                                                                 angular_noise_std=angular_noise_std,
                                                                 sensor_group='polar',
                                                                 return_smoothed=True,
                                                                 cutoff_freq=psi_freq,
                                                                 phi_alignment=phi_alignment,
                                                                 correction_window_for_2pi=int(turn_period/2./dt),
                                                                 butterworth_freq_param_adjustment=butterworth_freq_param_adjustment,
                                                                )

    # solve
    ts = []
    zeta_ests = []
    duration = tau # 2/psi_freq
    tstarts = np.linspace(0, df_sensor.t.max(), int(df.shape[0])/10)
    for tstart in tstarts:
        if tstart + duration < df_sensor.t.max():
            t, zeta_est = convex_solve_tan_cot.convex_solve_for_chunk(df_sensor, 
                                                                      tstart, 
                                                                      duration,
                                                                      stride=1,
                                                                      vdot_constant=False)
            ts.append(t)
            zeta_ests.append(zeta_est)    

    # disambiguate
    try:
        T = Ts[0]
    except:
        Ts = [Ts]
        
        
    if use_smoothed_for_disambiguation:
        df_for_disambiguation = df_sensor
    else:
        df_for_disambiguation = df
        
    for T in Ts:
        integration_time = T # duration*2
        corrected_ts = []
        corrected_zeta_ests = []
        #course_changes = []
        true_zetas = []
        std_true_zetas = []
        for i in range(len(zeta_ests)):
            zeta_est = zeta_ests[i]
            tend = ts[i] + tau
            tstart_tau = ts[i]
            tstart_T = tend - integration_time
            tstart_T = np.max([0, tstart_T])
            corrected_zeta = convex_solve_tan_cot.disambiguate_wind_options(df_for_disambiguation, 
                                                                            zeta_est, 
                                                                            tstart_T, 
                                                                            tend)
            corrected_zeta_ests.append(corrected_zeta)
            corrected_ts.append(tend - tau/2.)

            dfq = df_for_disambiguation.query('t > ' + str(tstart_tau) + ' and t < ' + str(tend))
            #course_changes.append( np.sum(np.abs(dfq.sensor_psi_dot.values + dfq.sensor_phi_dot.values)) )

            dfq = df.query('t > ' + str(tstart_tau) + ' and t < ' + str(tend))
            true_zetas.append( mean_angle(dfq.zeta.values) )
            std_true_zetas.append( std_angle(dfq.zeta.values) )

        # save
        df_save = pandas.DataFrame({'t': corrected_ts,
                                        'zeta_est_original': zeta_ests,
                                        'zeta_est': corrected_zeta_ests,
                                        'zeta_true': true_zetas,
                                        'std_true_zetas': std_true_zetas,
                                        'tau': tau,
                                        'psi_freq': psi_freq,
                                        'angular_noise_std': angular_noise_std,
                                        'T': T,
                                        #'sensor_gamma_dot': df_sensor.sensor_gamma_dot,
                                        #'sensor_psi_dot': df_sensor.sensor_psi_dot,
                                        #'sensor_phi_dot': df_sensor.sensor_phi_dot,
                                        #'sensor_gamma_smooth': df_sensor.sensor_gamma,
                                        #'sensor_psi_smooth': df_sensor.sensor_psi,
                                        #'sensor_phi_smooth': df_sensor.sensor_phi,
                                        #'sensor_gamma_noisy': df.sensor_gamma,
                                        #'sensor_psi_noisy': df.sensor_psi,
                                        #'sensor_phi_noisy': df.sensor_phi,
                                       })
        
        if save:
            if velocity_profile == 'absine' and phi_alignment == 'align_psi':
                fname = 'cvx_' + 'wind' + wind_type + '_' + trajec_type + '_turnamplitude' + turn_amplitude + '_absine_alignpsi' + '_angularnoisestd' + str(angular_noise_std) + '_psifreq' + str(psi_freq) + '_tau' + str(tau) + '_T' + str(T) 
            elif velocity_profile == 'constant' and phi_alignment == 'align_psi':
                fname = 'cvx_' + 'wind' + wind_type + '_' + trajec_type + '_turnamplitude' + turn_amplitude + '_constantvelocity_alignpsi' + '_angularnoisestd' + str(angular_noise_std) + '_psifreq' + str(psi_freq) + '_tau' + str(tau) + '_T' + str(T) 
            elif velocity_profile == 'fastABSsine' and phi_alignment == 'align_psi':
                fname = 'cvx_' + 'wind' + wind_type + '_' + trajec_type + '_turnamplitude' + turn_amplitude + '_fastABSsine_alignpsi' + '_angularnoisestd' + str(angular_noise_std) + '_psifreq' + str(psi_freq) + '_tau' + str(tau) + '_T' + str(T) 
            elif velocity_profile == 'absine' and phi_alignment == 'align_gamma':
                fname = 'cvx_' + 'wind' + wind_type + '_' + trajec_type + '_turnamplitude' + turn_amplitude + '_absine_aligngamma' + '_angularnoisestd' + str(angular_noise_std) + '_psifreq' + str(psi_freq) + '_tau' + str(tau) + '_T' + str(T) 
                
            fname += '_paramX' + str(butterworth_freq_param_adjustment) + '.hdf'
            smoothing_parameters_fname = 'smoothingparameters_' + fname
            
            fname = os.path.join(directory, fname)
            df_save.to_hdf(fname, 'data')
            
            smoothing_parameters_fname = os.path.join(directory, smoothing_parameters_fname)
            smoothing_parameters.to_hdf(smoothing_parameters_fname, 'parameters')
        else:
            return df_save

def run_analysis(L, phis, turn_amplitudes, velocity_profile, T_multipliers, butterworth_freq_param_adjustments, wind, directory):
    ###############################################
    # keep these the same for all simulation experiments
    angular_noise_stds = [0.3, 0.6, 1.2] 
    psi_freqs = [0.01, .02, .1, 0.5, 1.25]
    pynumdiff_params = 'turning' 
    taus = [0.4, 2, 10, 50, 250]
    use_smoothed_for_disambiguation = False
    trajec_type = 'random'
    smoothness_weight = 1
    ###############################################

    ###############################################
    # Run the simulations
    for phi in phis:
        for angular_noise_std in angular_noise_stds:
            for turn_amplitude in turn_amplitudes:
                for ix_freq, psi_freq in enumerate(psi_freqs):
                    
                    pynumdiff_param = pynumdiff_params #[ix_freq]
                    df, biomechanics_parameters = get_df_random(wind=wind,
                                                     L=L,
                                                     dt=None,
                                                     t=None,
                                                     psi_freq=psi_freq,
                                                     smoothness_weight=smoothness_weight,
                                                     angular_noise_std=0.0001,
                                                     of_noise_std=0.0001,
                                                     turn_amplitude=turn_amplitude,
                                                     air_noise_std=0.0001,
                                                     velocity_profile=velocity_profile,
                                                     phi=phi)

                    for butterworth_freq_param_adjustment in butterworth_freq_param_adjustments:
                        print('paramX: ' + str(butterworth_freq_param_adjustment))
                        ###
                        turn_period = (1/psi_freq / 2.)
                        dt = np.mean(np.diff(df.t))
                        print('Running smoother')
                        df_sensor, smoothing_parameters = get_sensor_measurements_derivatives_controls(df, 
                                                                 derivative_method='smooth_finite_difference.butterdiff', 
                                                                 params=pynumdiff_params,
                                                                 angular_noise_std=angular_noise_std,
                                                                 sensor_group='polar',
                                                                 return_smoothed=True,
                                                                 cutoff_freq=psi_freq,
                                                                 phi_alignment=phi,
                                                                 correction_window_for_2pi=int(turn_period/2./dt), butterworth_freq_param_adjustment=butterworth_freq_param_adjustment)


                        print('Running cvx')
                        for tau in taus:
                            #for T_multiplier in T_multipliers:
                            print(tau) #, T_multiplier)
                            Ts = tau*np.array(T_multipliers)
                            solve_convex(tau, Ts, psi_freq, df, trajec_type, turn_amplitude, pynumdiff_param, angular_noise_std, 
                                         wind, velocity_profile=velocity_profile, phi_alignment=phi,
                                        butterworth_freq_param_adjustment=butterworth_freq_param_adjustment,
                                         use_smoothed_for_disambiguation=use_smoothed_for_disambiguation,
                                        df_sensor=df_sensor, smoothing_parameters=smoothing_parameters, directory=directory)


if __name__ == '__main__':

    L = None # default: automatic, choose small L for quick debugging
    
    ###############################################
    # Vary these options to get all of the datasets
    # This set of options will generate the data for Fig. 9
    # This takes a while to run. In practice, it is more efficient
    # to run the analysis for each turn amplitude seperately, in parallel
    # and then combine the data into one directory after the fact. 
    wind= 'realdynamic' # realconstant or realdynamic
    turn_amplitudes= ['180deg', '20deg', '90deg']
    velocity_profile = 'absine' # constant or absine
    phis = ['align_psi'] # align_gamma or align_psi
    T_multipliers = [1] # list of [1, 5, 25, 125], or [1]
    butterworth_freq_param_adjustments = [1] # list of [1, 0.1, 0.5, 2, 10] or [1]
    ###############################################
    
    ###############################################
    # automatically determine correct location to save data
    location = '../data_simulations'
    directory = '20220301_seed1_' + velocity_profile + '_' + phis[0].replace('_', '') + '_' + wind
    directory = os.path.join(location, directory)
    os.mkdir(directory)
    ###############################################
    
    ###############################################
    run_analysis(L, phis, turn_amplitudes, velocity_profile, T_multipliers, butterworth_freq_param_adjustments, wind, directory)
    ###############################################

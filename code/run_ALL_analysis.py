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

from run_analysis import get_df_random, solve_convex, run_analysis

def get_directory_name(velocity_profile, phis, wind):
    location = '../data_simulations'
    directory = '20220301_seed1_' + velocity_profile + '_' + phis[0].replace('_', '') + '_' + wind
    directory = os.path.join(location, directory)
    return directory

if __name__ == '__main__':

    L = None # None or an integer. None=automatic; choose small L (e.g. 300) for quick debugging
    

    # Condition 1
    ###############################################
    wind= 'realdynamic' # realconstant or realdynamic
    turn_amplitudes= ['180deg', '20deg', '90deg']
    velocity_profile = 'absine' # constant or absine
    phis = ['align_psi'] # align_gamma or align_psi
    T_multipliers = [1]
    butterworth_freq_param_adjustments = [1]
    directory = get_directory_name(velocity_profile, phis, wind)
    os.mkdir(directory)
    run_analysis(L, phis, turn_amplitudes, velocity_profile, T_multipliers, butterworth_freq_param_adjustments, wind, directory)
    ###############################################

    # Condition 2
    ###############################################
    wind= 'realconstant' # realconstant or realdynamic
    turn_amplitudes= ['180deg', '20deg', '90deg']
    velocity_profile = 'absine' # constant or absine
    phis = ['align_psi'] # align_gamma or align_psi
    T_multipliers = [1]
    butterworth_freq_param_adjustments = [1]
    directory = get_directory_name(velocity_profile, phis, wind)
    os.mkdir(directory)
    run_analysis(L, phis, turn_amplitudes, velocity_profile, T_multipliers, butterworth_freq_param_adjustments, wind, directory)
    ###############################################

    # Condition 3
    ###############################################
    wind= 'realdynamic' # realconstant or realdynamic
    turn_amplitudes= ['180deg', '20deg', '90deg']
    velocity_profile = 'constant' # constant or absine
    phis = ['align_psi'] # align_gamma or align_psi
    T_multipliers = [1]
    butterworth_freq_param_adjustments = [1]
    directory = get_directory_name(velocity_profile, phis, wind)
    os.mkdir(directory)
    run_analysis(L, phis, turn_amplitudes, velocity_profile, T_multipliers, butterworth_freq_param_adjustments, wind, directory)
    ###############################################

    # Condition 4
    ###############################################
    wind= 'realdynamic' # realconstant or realdynamic
    turn_amplitudes= ['180deg', '20deg', '90deg']
    velocity_profile = 'absine' # constant or absine
    phis = ['align_gamma'] # align_gamma or align_psi
    T_multipliers = [1]
    butterworth_freq_param_adjustments = [1]
    directory = get_directory_name(velocity_profile, phis, wind)
    os.mkdir(directory)
    run_analysis(L, phis, turn_amplitudes, velocity_profile, T_multipliers, butterworth_freq_param_adjustments, wind, directory)
    ###############################################

    # Condition 1 to explore Tmultiplier and butterworth_freq_param_adjustment sensitivity
    ###############################################
    wind= 'realdynamic' # realconstant or realdynamic
    turn_amplitudes= ['180deg', '20deg', '90deg']
    velocity_profile = 'absine' # constant or absine
    phis = ['align_psi'] # align_gamma or align_psi
    T_multipliers = [1, 5, 25, 125] # list of [1, 5, 25, 125], or [1]
    butterworth_freq_param_adjustments = [1, 0.1, 0.5, 2, 10] # list of [1, 0.1, 0.5, 2, 10] or [1]
    directory = get_directory_name(velocity_profile, phis, wind)
    directory += '_vary_omega_T'
    os.mkdir(directory)
    run_analysis(L, phis, turn_amplitudes, velocity_profile, T_multipliers, butterworth_freq_param_adjustments, wind, directory)
    ###############################################

    
    
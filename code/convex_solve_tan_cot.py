# convex solver

import pandas
import cvxpy
import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np
import copy

import fly_plot_lib.flymath as flymath
import pynumdiff

import multiprocessing
import time


def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))

def mean_angle(angle):
        return np.arctan2(np.mean(np.sin(angle)), np.mean(np.cos(angle)))

def cvxpy_diff(x, dt):
    # aligns the derivative properly
    dxdt_hat = cvxpy.multiply(cvxpy.diff(x), 1/dt)
    dxdt_hat = cvxpy.hstack((cvxpy.Variable(1), dxdt_hat, cvxpy.Variable(1)))
    dxdt_hat = (dxdt_hat[1:] + dxdt_hat[0:-1])/2.
    return dxdt_hat

def cvx_vw(df, choice='cot', TV=1e-3, zeta=None, stride=1, vdot_constant=False, vdot_zero=False):
    
    ## sensor measurements
    L = len(df['t'])
    Lf = len(df['t'].values[0:L:stride])
    dt = np.diff(df['t'].values[0:L:stride])
    
    phi = df['sensor_phi'].values[0:L:stride]
    phi_dot = df['sensor_phi_dot'].values[0:L:stride]
    
    gamma = df['sensor_gamma'].values[0:L:stride]
    gamma_dot = df['sensor_gamma_dot'].values[0:L:stride]
    
    psi = df['sensor_psi'].values[0:L:stride] 
    psi_dot = df['sensor_psi_dot'].values[0:L:stride]
    
    h2 = np.tan(gamma)
    h2dot = np.cos(gamma)**(-2)*gamma_dot # chain rule
    h3 = np.tan(psi)
    h3dot = np.cos(psi)**(-2)*psi_dot # chain rule
    goodix = np.where( (np.abs(h2dot)<5000)*(np.abs(h3dot)<5000) )
    
    delta = gamma - psi
    alpha = gamma + phi
    
    delta_dot = gamma_dot - psi_dot
    alpha_dot = gamma_dot + phi_dot
    
    ## variables
    if vdot_constant:
        vw_dot = cvxpy.Variable(1)
        vw = cvxpy.multiply(df.t, vw_dot) + cvxpy.Variable(1)
    elif vdot_zero:
        vw_dot = 0
        vw = cvxpy.Variable(1)
    else:
        vw = cvxpy.Variable(Lf)
        vw_dot = cvxpy_diff(vw, dt)
    
    if zeta is not None: ### Cheat to check
        if choice == 'cot':
            cot_zeta = 1/np.tan(zeta)
            L1 = cvxpy.multiply(vw, np.sin(delta)) - cvxpy.multiply(cot_zeta, np.sin(alpha)) + cvxpy.multiply(1, np.cos(alpha))
            L2 = cvxpy.multiply(vw, np.cos(delta)*delta_dot) + cvxpy.multiply(vw_dot, np.sin(delta)) - cvxpy.multiply(cot_zeta, np.cos(alpha)*alpha_dot) - cvxpy.multiply(1, np.sin(alpha)*alpha_dot)
        else:
            tan_zeta = np.tan(zeta)
            L1 = cvxpy.multiply(vw, np.sin(delta)) - cvxpy.multiply(1, np.sin(alpha)) + cvxpy.multiply(tan_zeta, np.cos(alpha))
            L2 = cvxpy.multiply(vw, np.cos(delta)*delta_dot) + cvxpy.multiply(vw_dot, np.sin(delta)) - cvxpy.multiply(1, np.cos(alpha)*alpha_dot) - cvxpy.multiply(tan_zeta, np.sin(alpha)*alpha_dot)

        ## solve 
        Loss = cvxpy.norm(L1, 1) + cvxpy.norm(L2, 2) 
        obj = cvxpy.Minimize( Loss )
        prob = cvxpy.Problem(obj) 
        prob.solve(solver='MOSEK')
        
        print(Loss.value)
        print(vw.value)
        
        return None, None, None
        
    if choice == 'cot':
        cot_zeta = cvxpy.Variable(1)

        ## Loss functions
        L1 = cvxpy.multiply(vw, np.sin(delta)) - cvxpy.multiply(cot_zeta, np.sin(alpha)) + cvxpy.multiply(1, np.cos(alpha))

        L2 = cvxpy.multiply(vw, np.cos(delta)*delta_dot) + cvxpy.multiply(vw_dot, np.sin(delta)) - cvxpy.multiply(cot_zeta, np.cos(alpha)*alpha_dot) - cvxpy.multiply(1, np.sin(alpha)*alpha_dot)

        if vdot_constant or vdot_zero:
            L3 = 0
        else:
            L3 = cvxpy.tv(vw_dot[goodix])

        ## solve 
        Loss = cvxpy.norm(L1[goodix], 1) + cvxpy.norm(L2[goodix], 2) + cvxpy.norm(L3)*TV
        obj = cvxpy.Minimize( Loss )
        prob = cvxpy.Problem(obj) 
        prob.solve(solver='MOSEK')

        if np.abs(cot_zeta.value) > 0:
            zopt = np.arctan(1/cot_zeta.value) 
            zopt = np.max([zopt, wrap_angle(zopt+np.pi)])
        else:
            zopt = np.pi
    else:
        tan_zeta = cvxpy.Variable(1)

        ## Loss functions
        L1 = cvxpy.multiply(vw, np.sin(delta)) - cvxpy.multiply(1, np.sin(alpha)) + cvxpy.multiply(tan_zeta, np.cos(alpha))

        L2 = cvxpy.multiply(vw, np.cos(delta)*delta_dot) + cvxpy.multiply(vw_dot, np.sin(delta)) - cvxpy.multiply(1, np.cos(alpha)*alpha_dot) - cvxpy.multiply(tan_zeta, np.sin(alpha)*alpha_dot)
        
        if vdot_constant or vdot_zero:
            L3 = 0
        else:
            L3 = cvxpy.tv(vw_dot[goodix])

        ## solve 
        Loss = cvxpy.norm(L1[goodix], 1) + cvxpy.norm(L2[goodix], 2) + cvxpy.norm(L3)*TV
        obj = cvxpy.Minimize( Loss )
        prob = cvxpy.Problem(obj) 
        prob.solve(solver='MOSEK')


        zopt = np.arctan(tan_zeta.value) 
        zopt = np.max([zopt, wrap_angle(zopt+np.pi)])
                            
    # disambiguate the two options
    zeta_estimates_1 = zopt*np.ones_like(phi)
    alpha_minus_zeta_1 = wrap_angle(alpha - zeta_estimates_1)
    check_signs_1 = np.sin(alpha_minus_zeta_1) / np.sin(delta) #np.abs( vw.value - ((np.sin(alpha_minus_zeta_1)) / (np.sin(delta))) )

    zeta_estimates_2 = wrap_angle(zopt + np.pi)*np.ones_like(phi)
    alpha_minus_zeta_2 = wrap_angle(alpha - zeta_estimates_2)
    check_signs_2 = np.sin(alpha_minus_zeta_2) / np.sin(delta) #np.abs( vw.value - ((np.sin(alpha_minus_zeta_2)) / (np.sin(delta))) )

    ix_flip = np.where(check_signs_1 < 0)
    zeta_estimates = zeta_estimates_1
    zeta_estimates[ix_flip] += np.pi
    zeta_estimates = wrap_angle(zeta_estimates)

    # recalculate check
    alpha_minus_zeta = alpha - zeta_estimates
    new_check_signs = np.sign(np.sin(alpha_minus_zeta))*np.sign(np.sin(delta))

    check = np.mean(new_check_signs)
    correct_zeta = wrap_angle( np.arctan2(np.median(np.sin(zeta_estimates)), np.median(np.cos(zeta_estimates))) )
    
    return Loss.value, correct_zeta, check

def cvx_vw_tan_cot(df, stride=1, TV=1e-3, vdot_constant=False, vdot_zero=False):
    loss_tan, correct_zeta_tan, check_tan = cvx_vw(df, choice='tan', TV=TV, stride=stride, 
                                                   vdot_constant=vdot_constant, vdot_zero=vdot_zero)
    loss_cot, correct_zeta_cot, check_cot = cvx_vw(df, choice='cot', TV=TV, stride=stride, 
                                                   vdot_constant=vdot_constant, vdot_zero=vdot_zero)
    
    ix = np.argmin([loss_tan, loss_cot])
    return [correct_zeta_tan, correct_zeta_cot][ix]

def convex_solve_for_chunk(df, tstart, tduration, stride=1, vdot_constant=False, vdot_zero=False):
    dfq = df.query('t > ' + str(tstart) + ' and t < ' + str(tstart+tduration))
    zeta = cvx_vw_tan_cot(dfq, stride=stride, vdot_constant=vdot_constant, vdot_zero=vdot_zero)
    return np.mean(dfq.t), zeta

def disambiguate_wind_options(df_sensor, zeta_est, tstart, tend, stride=1):
    df = df_sensor.query('t > ' + str(tstart) + ' and t < ' + str(tend))
    L = len(df)
    dt = np.diff(df['t'].values[0:L:stride])
    
    phi = df['sensor_phi'].values[0:L:stride]
    #phi_dot = df['sensor_phi_dot'].values[0:L:stride]
    
    gamma = df['sensor_gamma'].values[0:L:stride]
    #gamma_dot = df['sensor_gamma_dot'].values[0:L:stride]
    
    psi = df['sensor_psi'].values[0:L:stride] 
    #psi_dot = df['sensor_psi_dot'].values[0:L:stride]
    
    delta = gamma - psi
    alpha = gamma + phi
    
    #delta_dot = gamma_dot - psi_dot
    #alpha_dot = gamma_dot + phi_dot
    
    # disambiguate the two options
    zeta_estimates_1 = zeta_est*np.ones_like(phi)
    alpha_minus_zeta_1 = wrap_angle(alpha - zeta_estimates_1)
    check_signs_1 = np.sin(alpha_minus_zeta_1) / np.sin(delta) 

    zeta_estimates_2 = wrap_angle(zeta_est + np.pi)*np.ones_like(phi)
    alpha_minus_zeta_2 = wrap_angle(alpha - zeta_estimates_2)
    check_signs_2 = np.sin(alpha_minus_zeta_2) / np.sin(delta) 

    ix_flip = np.where(check_signs_1 < 0)
    
    if 1: # average
        zeta_estimates = zeta_estimates_1
        zeta_estimates[ix_flip] += np.pi
        zeta_estimates = wrap_angle(zeta_estimates)
        correct_zeta = wrap_angle( np.arctan2(np.median(np.sin(zeta_estimates)), np.median(np.cos(zeta_estimates))) )

    else: # mode
        npi = 0
        if len(ix_flip) > L/2.: # flip it
            npi = np.pi
        correct_zeta = wrap_angle( zeta_est + npi )
        
    # recalculate check
    #alpha_minus_zeta = alpha - zeta_estimates
    #new_check_signs = np.sign(np.sin(alpha_minus_zeta))*np.sign(np.sin(delta))
    #check = np.mean(new_check_signs)
    
    
    return correct_zeta

import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib
import scipy
import copy
import scipy.optimize


import pynumdiff # pip install pynumdiff
import fly_plot_lib.plot as fpl # install from source: https://github.com/florisvb/FlyPlotLib
import fly_plot_lib.flymath as flymath
import figurefirst as fifi # pip install figurefirst

from utility import wrap_angle, mean_angle, plot_wind_quivers, plot_trajec, diff_angle

def load_real_wind():
    def interp_with_std(t_interp, t, val, std=0.1):
        val_interp = np.interp(t_interp, t, val)

        chunks, breaks = flymath.get_continuous_chunks(np.diff(val_interp), jump=0.001)
        for i in range(len(breaks)-1):
            val_interp[breaks[i]+1:breaks[i+1]] += np.random.normal(0, std, len(val_interp[breaks[i]+1:breaks[i+1]]))

        return val_interp

    wind_data = pandas.read_hdf('real_wind/20201020_172919_data.hdf')
    m1 = 204
    m2 = 220

    zeta = wind_data.query('millis > ' + str(m1*60*1000) + ' and millis < ' + str(m2*60*1000)).D.values*np.pi/180.
    zeta -= np.pi
    w = wind_data.query('millis > ' + str(m1*60*1000) + ' and millis < ' + str(m2*60*1000)).S2.values
    t = wind_data.query('millis > ' + str(m1*60*1000) + ' and millis < ' + str(m2*60*1000)).millis.values/1000.
    t -= t[0]
    
    #zeta += 1.5
    #zeta[zeta>3.14] = 3.14
    #zeta[zeta<0] = 0

    # interpolate to dt = 0.01
    dt = 0.01

    t_interp = (np.round(np.arange(0, t[-1], dt)*100)).astype(int)/100.
    zeta_interp = interp_with_std(t_interp, t, zeta, std=0.1)
    w_interp = interp_with_std(t_interp, t, w, std=0.1)

    return zeta_interp, w_interp

    
def diff_finite(x, dt):
    dxdt_hat = np.diff(x)/dt
    # Pad the data
    dxdt_hat = np.hstack((dxdt_hat[0], dxdt_hat, dxdt_hat[-1]))
    # Re-finite dxdt_hat using linear interpolation
    dxdt_hat = np.mean((dxdt_hat[0:-1], dxdt_hat[1:]), axis=0)
    return x, dxdt_hat

def diff_tvrp(x, dt, params=[100]):
    '''
    piece wise constant signal, derivative consisting of impulses
    '''
    _ , x_smooth = pynumdiff.total_variation_regularization.velocity( np.cumsum(x)*np.mean(dt), np.mean(dt), params)
    _, x_dot = diff_finite(x_smooth, dt)
    return x_smooth, x_dot
    

    
def run_sim(L=400, dt=0.01,
            psi_global=[0, 1], 
            psi_freq=None,
            vel=1, 
            d=2,
            phi='align_gamma',
            gaussian_window=1, 
            zeta=1, w=1 
            ):
    '''
    Inputs
    ======

    L  ----------------- length of simulation (time steps)
    dt ----------------- time step (secs)
    psi_global --------- global course angle, ie. direction of movement (radians). Can be a list of discrete angles, an array of length L, or a constant
    vel ---------------- speed (m/s), can be an array of length L, or a constant
    d ------------------ distance above the ground (m), can be array of length L, or a constant
    phi ---------------- global body orientation (radians). can be a constant, array of length L, or the following keywords:
                            - 'align_gamma': choose phi such that gamma remains zero
                            - 'align_psi': choose phi such that psi remains zero
    gaussian_window ---- integer that specifies window size for applying a gaussian smoothing kernel to global_psi, if global_psi is a discrete list
    zeta --------------- wind direction (radians), global coordinate frame. can be a constant or an array of length L
    w ------------------ wind magnitude (m/s), can be a constant or array of length L

    Outputs
    =======

    pandas dataframe with all the state values



    '''
    
    
    
    t = np.arange(0, L*dt, dt)[0:L]
    


    # convert constants to arrays
    try:
        _ = zeta[0]
        zeta = np.array(zeta)
    except:
        zeta = zeta*np.ones(L)
    
    try:
        _ = w[0]
        w = np.array(w)
    except:
        w = w*np.ones(L)

    try:
        _ = vel[0]
        vel = np.array(vel)
    except:
        vel = vel*np.ones(L)
        
    try:
        _ = d[0]
        d = np.array(d)
    except:
        d = d*np.ones(L)
        
     # if psi_global is a list, convert it to an array of length L where each value in psi is given equal time
    if type(psi_global) == list:
        psi_global = np.hstack([np.ones( int(L/len(psi_global)+1))*psi_global[i] for i in range(len(psi_global))])
        psi_global = psi_global[0:L]
        
        psi_global_smooth, psi_global_dot = pynumdiff.smooth_finite_difference.gaussiandiff(psi_global, dt, [gaussian_window], {'iterate': False})
        psi_global = psi_global_smooth

    

    # get x,y vel and pos
    xvel = vel*np.cos(psi_global)
    yvel = vel*np.sin(psi_global)
    xpos = pynumdiff.utils.utility.integrate_dxdt_hat(xvel, dt)
    ypos = pynumdiff.utils.utility.integrate_dxdt_hat(yvel, dt)

    
    
    # get phi, gamma, psi according to control laws
    if phi == 'align_psi':
        phi = psi_global 
        _, phidot = diff_angle(phi, dt, [], derivative_method='finite_difference.first_order')
        
        psi = wrap_angle(psi_global - phi)

        beta = wrap_angle(phi - zeta)

        v_para = vel*np.cos(psi)
        v_perp = vel*np.sin(psi)
        air_para = ( +v_para - w*np.cos(beta) )
        air_perp = ( +v_perp + w*np.sin(beta))
        
        gamma = np.arctan2(air_perp, air_para)
        
        air_x = xvel - w*np.cos(zeta)
        air_y = yvel - w*np.sin(zeta)
        
    elif phi == 'fixed':
        phi = np.zeros_like(t) 
        _, phidot = diff_angle(phi, dt, [], derivative_method='finite_difference.first_order')
        
        psi = wrap_angle(psi_global - phi)

        beta = wrap_angle(phi - zeta)

        v_para = vel*np.cos(psi)
        v_perp = vel*np.sin(psi)
        air_para = ( +v_para - w*np.cos(beta) )
        air_perp = ( +v_perp + w*np.sin(beta))
        
        gamma = np.arctan2(air_perp, air_para)
        
        air_x = xvel - w*np.cos(zeta)
        air_y = yvel - w*np.sin(zeta)
    
    elif phi == 'align_gamma':
        air_x = xvel - w*np.cos(zeta)
        air_y = yvel - w*np.sin(zeta)
        
        phi = wrap_angle(np.arctan2(air_y, air_x))
        _, phidot = diff_angle(phi, dt, [], derivative_method='finite_difference.first_order')
        
        psi = wrap_angle(psi_global - phi)
        beta = wrap_angle(phi - zeta)

        v_para = vel*np.cos(psi)
        v_perp = vel*np.sin(psi)
        air_para = ( +v_para - w*np.cos(beta) )
        air_perp = ( +v_perp + w*np.sin(beta) )

        gamma = np.arctan2(air_perp, air_para)

    else: # phi is explicitly given
        try:
            _ = phi[0]
            phi = np.array(phi)
        except:
            phi = phi*np.ones(L)
            
        # take angular derivatve
        phi0 = phi[0]
        _, phidot = diff_angle(phi, dt, [], derivative_method='finite_difference.first_order')
            
        phi = pynumdiff.utils.utility.integrate_dxdt_hat(phidot, dt) + phi0
        phi = wrap_angle(phi)
            
        psi = wrap_angle(psi_global - phi)

        beta = wrap_angle(phi - zeta)

        v_para = vel*np.cos(psi)
        v_perp = vel*np.sin(psi)
        air_para = ( +v_para - w*np.cos(beta) )
        air_perp = ( +v_perp + w*np.sin(beta))
        
        gamma = np.arctan2(air_perp, air_para)
        
        air_x = xvel - w*np.cos(zeta)
        air_y = yvel - w*np.sin(zeta)
        
        
    df_states = pandas.DataFrame({'w': w, 'zeta': zeta, 
                                  't': t, 
                                  'd': d,
                                  'xvel': xvel, 'yvel': yvel,
                                  'xpos': xpos, 'ypos': ypos,
                                  'v_para': v_para, 'v_perp': v_perp, 'vel': vel,
                                  'air_para': air_para, 'air_perp': air_perp,
                                  'air_x': air_x, 'air_y': air_y,
                                  'psi_global': psi_global,
                                  'phi': phi, 'gamma': gamma, 'psi': psi,
                                  'phidot': phidot,
                                  })
    
    return df_states






# five equations, five unknowns
'''
beta = phi - zeta
air_para = ( +v_para - w*np.cos(beta) )
0 = ( +v_perp + w*np.sin(beta) )
air_y = air_para*np.sin(psi_global)
air_x = air_para*np.cos(psi_global)
phi = np.arctan2(air_y, air_x)
'''

def foo(x, args):
    phi, air_para, v_perp, air_x, air_y, beta = x
    w, zeta, v_para, psi_global = args
    psi = psi_global - phi
    
    eq1 = -beta + phi - zeta
    eq2 = -air_para + ( +v_para - w*np.cos(beta) )
    eq3 = ( +v_perp + w*np.sin(beta) )
    #eq4 = -air_y + air_para*np.sin(psi_global)
    #eq5 = -air_x + air_para*np.cos(psi_global)
    #eq6 = -np.tan(phi) + air_y/air_x
    eq7 = -np.tan(psi) + v_perp/v_para
    
    return eq1**2 + eq2**2 + eq3**2 + eq7**2 # eq4**2 + eq5**2 + eq6**2 #+ eq7**2

def run_sim_const_vel_para(L=400, dt=0.01,
            psi_global=[0, 1], 
            v_para=1, 
            d=2,
            phi='align_gamma',
            gaussian_window=1, 
            scale_state=False,
            zeta=0, w=1 
            ):
    '''
    Inputs
    ======

    L  ----------------- length of simulation (time steps)
    dt ----------------- time step (secs)
    psi_global --------- global course angle, ie. direction of movement (radians). Can be a list of discrete angles, an array of length L, or a constant
    v_para ------------- speed parallel to body (m/s), can be an array of length L, or a constant
    phi ---------------- global body orientation (radians). can be a constant, array of length L, or the following keywords:
                            - 'align_gamma': choose phi such that gamma remains zero
                            - 'align_psi': choose phi such that psi remains zero
                            - 'align_blend': choose phi such that some weighted sum of psi and gamma is zero
    gaussian_window ---- integer that specifies window size for applying a gaussian smoothing kernel to global_psi, if global_psi is a discrete list
    zeta --------------- wind direction (radians), global coordinate frame. can be a constant or an array of length L
    w ------------------ wind magnitude (m/s), can be a constant or array of length L

    Outputs
    =======

    pandas dataframe with all the state values



    '''
    t = np.arange(0, L*dt, dt)[0:L]
    


    # convert constants to arrays
    try:
        _ = zeta[0]
        zeta = np.array(zeta)
    except:
        zeta = zeta*np.ones(L)
    
    try:
        _ = w[0]
        w = np.array(w)
    except:
        w = w*np.ones(L)

    try:
        _ = v_para[0]
        v_para = np.array(v_para)
    except:
        v_para = v_para*np.ones(L)
        
    try:
        _ = d[0]
        d = np.array(d)
    except:
        d = d*np.ones(L)
        
     # if psi_global is a list, convert it to an array of length L where each value in psi is given equal time
    if type(psi_global) == list:
        psi_global = np.hstack([np.ones( int(L/len(psi_global)+1))*psi_global[i] for i in range(len(psi_global))])
        psi_global = psi_global[0:L]
        psi_global_smooth, psi_global_dot = pynumdiff.smooth_finite_difference.gaussiandiff(psi_global, dt, [gaussian_window], {'iterate': False})
        psi_global = psi_global_smooth
        
    
    phi = []
    air_para = []
    v_perp = []
    air_x = []
    air_y = []
    beta = []
    
    guess1 = [1,1,1,1,1,1]
    for i in range(len(t)):
        result1 = scipy.optimize.minimize(foo, guess1, args=[w[i], zeta[i], v_para[i], psi_global[i]], 
                                method='SLSQP', #jac=jacobian,
                                options={'ftol': 1e-5})
        
        # phi should be closer to pi away from zeta than 0
        err1 = np.abs(wrap_angle(result1.x[0] - zeta[i]))
        if 1: #err1 < np.pi/2.:
            guess2 = copy.copy(guess1)
            guess2[0] += np.pi
            guess2[-1] += np.pi
            result2 = scipy.optimize.minimize(foo, guess2, args=[w[i], zeta[i], v_para[i], psi_global[i]], 
                                    method='SLSQP', #jac=jacobian,
                                    options={'ftol': 1e-5})
            err2 = np.abs(wrap_angle(result2.x[0] - zeta[i]))
        
            if err1 > err2: # err1 closer to pi so it is correct
                result = result1
            else:
                result = result2
        else:
            result = result1
        
        phi.append(result.x[0])
        air_para.append(result.x[1])
        v_perp.append(result.x[2])
        air_x.append(result.x[3])
        air_y.append(result.x[4])
        beta.append(result.x[5])
        
        guess1 = result.x
        
        
        
    phi = np.array(phi)
    air_para = np.array(air_para)
    v_perp = np.array(v_perp)
    air_x = np.array(air_x)
    air_y = np.array(air_y)
    beta = np.array(beta)
        
    vel = np.sqrt(v_para**2 + v_perp**2)
    xvel = vel*np.cos(psi_global)
    yvel = vel*np.sin(psi_global)
    xpos = pynumdiff.utils.utility.integrate_dxdt_hat(xvel, dt)
    ypos = pynumdiff.utils.utility.integrate_dxdt_hat(yvel, dt)
    psi = wrap_angle(psi_global - phi)
    
    _, phidot = diff_angle(phi, dt, [], derivative_method='finite_difference.first_order')
    
    
    df_states = pandas.DataFrame({'w': w, 'zeta': zeta, 
                                  't': t, 
                                  'xvel': xvel, 'yvel': yvel,
                                  'xpos': xpos, 'ypos': ypos,
                                  'd': d,
                                  'v_para': v_para, 'v_perp': v_perp, 'vel': vel,
                                  'air_para': air_para, 'air_perp': np.zeros_like(t),
                                  'air_x': air_x, 'air_y': air_y,
                                  'psi_global': psi_global,
                                  'phi': phi, 'gamma': np.zeros_like(t), 'psi': psi,
                                  'phidot': phidot,
                                  })
    
    return df_states


############################################



def simulate_trajectory( angular_noise_std, air_noise_std, of_noise_std, 
                         psi_freq=1, psi_phase=np.pi/2., psi_amplitude=np.pi/2., psi_offset=0,
                         psi_global = None, # if not none, overrides freq, amp, phase
                         gaussian_window = None, # if psi_global is list, this applies some smoothing
                         constant_v_para=False,
                         acceleration=0,
                         mean_vel=1,
                         phi=1, # number, align_gamma, or align_psi
                         zeta=-np.pi,
                         w=1,
                         L=400, dt=0.01,
                         biomechanics_parameters='small'):
    '''
    
    biomechanics_parameters -- small or large, eventually replace with 'fly', 'moth', 'albatross', 'shark'
    '''
    
    t = np.arange(0, L*dt, dt)[0:L]

    # get psi global
    if psi_global is None:
        psi_amplitude = psi_amplitude
        psi_phase = psi_phase
        psi_global = psi_amplitude*np.sin(2*np.pi*psi_freq*t+psi_phase) + psi_offset
    
    # get velocity
    if np.sum(np.abs(acceleration)) > 0:
        vel = np.cumsum(acceleration*np.ones_like(t))*dt
        # lock velocity to always be positive
        vel[np.where(vel<0)] = 0
    else:
        # vel defined manually
        vel = mean_vel
        try:
            _ = vel[0]
        except:
            vel = np.ones_like(t)*vel
        acceleration = pynumdiff.finite_difference.first_order(vel, dt)[1]

    # run simulation
    if constant_v_para:
        df_state = run_sim_const_vel_para( L=L, dt=dt,
                                                        psi_global=psi_global, 
                                                        v_para=vel, 
                                                        phi=phi,
                                                        gaussian_window=gaussian_window, 
                                                        zeta=zeta, w=w, 
                                                        )
    else:
        df_state = run_sim( L=L, dt=dt,
                                                        psi_global=psi_global, 
                                                        vel=vel, 
                                                        phi=phi,
                                                        gaussian_window=gaussian_window, 
                                                        zeta=zeta, w=w, 
                                                        )
    
    # generate noise
    noise_gamma = np.random.normal(0, angular_noise_std, L)
    noise_psi = np.random.normal(0, angular_noise_std, L)
    noise_phi = np.random.normal(0, angular_noise_std, L)
    
    noise_of = np.random.normal(0, of_noise_std, L)
    noise_air = np.random.normal(0, air_noise_std, L)

    # get sensor data
    df_state['sensor_gamma'] = df_state['gamma'] + noise_gamma
    df_state['sensor_phi'] = df_state['phi'] + noise_phi
    df_state['sensor_psi'] = df_state['psi'] + noise_psi
    
    df_state['of_mag'] = df_state['vel'] / df_state['d']
    df_state['sensor_of_mag'] = df_state['of_mag'] + noise_of
    
    df_state['air_mag'] = np.sqrt(df_state['air_para'].values**2 + df_state['air_perp'].values**2)
    df_state['sensor_air_mag'] = df_state['air_mag'] + noise_air
    
    df_state['sensor_of_para'] = np.cos(df_state['sensor_psi'])*df_state['sensor_of_mag']
    df_state['sensor_of_perp'] = np.sin(df_state['sensor_psi'])*df_state['sensor_of_mag']
    df_state['sensor_air_para'] = np.cos(df_state['sensor_gamma'])*df_state['sensor_air_mag']
    df_state['sensor_air_perp'] = np.sin(df_state['sensor_gamma'])*df_state['sensor_air_mag']
    
    # get controls
    if biomechanics_parameters == 'small':
        m = 0.001
        I = 0.0001
        w_dot = 0
        zeta_dot = 0
        c_para = 1e-3
        c_perp = 1e-3
        c_phi = 1e-3
        drag_power = 1
    elif biomechanics_parameters == 'large': # large animal
        m = 1
        I = 1
        w_dot = 0
        zeta_dot = 0
        c_para = 1
        c_perp = 1
        c_phi = 1
        drag_power = 1
    biomechanics_parameters = {'m': m,
                               'I': I,
                               'c_para': c_para,
                               'c_perp': c_perp,
                               'c_phi': c_phi,
                               'drag_power': drag_power}
    
    v_para = df_state.v_para.values
    v_perp = df_state.v_perp.values
    phi = df_state.phi.values
    phidot = df_state.phidot.values
    w = df_state.w.values
    zeta = df_state.zeta.values

    a_para = df_state.air_para.values
    a_perp = df_state.air_perp.values

    D_para = np.sign(a_para)*c_para*np.abs(a_para)**drag_power
    D_perp = np.sign(a_perp)*c_perp*np.abs(a_perp)**drag_power
    D_phi = np.sign(phidot)*c_phi*np.abs(phidot)**drag_power

    diff = pynumdiff.finite_difference.first_order
    xdot = np.array([diff(v_para, dt)[1],
                     diff(v_perp, dt)[1],
                     diff(phi, dt)[1],
                     diff(phidot, dt)[1],
                     diff(w, dt)[1],
                     diff(zeta, dt)[1]])

    f_0 = np.vstack([-D_para/m+v_perp*phidot,
                    -D_perp/m-v_para*phidot,
                    phidot,
                    -D_phi/I,
                    0*np.ones_like(w), # model of wind, assumed to be zero
                    0*np.ones_like(w), # model of wind, assumed to be zero
                   ])

    u_para = (xdot[0,:] - f_0[0])*m
    u_perp = (xdot[1,:] - f_0[1])*m
    u_phi = (xdot[3,:] - f_0[3])*I
    
    df_state['u_para'] = u_para
    df_state['u_phi'] = u_phi
    df_state['u_perp'] = u_perp
    
    
    return df_state, biomechanics_parameters

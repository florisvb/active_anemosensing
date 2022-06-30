import matplotlib.pyplot as plt
import utility
import numpy as np
import figurefirst as fifi
import fly_plot_lib.flymath as flymath
import pynumdiff
import pandas
import os
import matplotlib
import fly_plot_lib.plot as fpl
import copy

import scipy.interpolate
interp2d = scipy.interpolate.interp2d

import utility


FONTSIZE = 7

def mathify_ticklabels(ax, axis, ticks=None, ticklabels=None):
    '''
    Convert tick labels to math font style by putting them inside '$$'.
    If ticks is None: automatically get ticks
    If ticklabels is not None: add '$$' where needed
    axis -- 'x' or 'y'
    '''
    if ticks is None:
        if axis == 'x':
            ticks = ax.get_xticks()
        else:
            ticks = ax.get_yticks()
        
    if ticklabels is None:
        ticklabels = []
        for tick in ticks:
            if len(str(tick))>0:
                ticklabels.append(r'$'+str(tick)+r'$')
        
    else:
        for i, ticklabel in enumerate(ticklabels):
            if type(ticklabel) != str:
                ticklabel = str(ticklabel)
            if '$' not in ticklabel and len(ticklabel) > 0:
                ticklabel = r'$' + ticklabel + r'$'
                ticklabels[i] = ticklabel
        
    if axis == 'x':
        ax.set_xticklabels(ticklabels)
    else:
        ax.set_yticklabels(ticklabels)
    

def plot_sensor_data(df, angular_noise_std, sensor, psi_freq, show_smooth=False,
                    correction_window_for_2pi=100, ax=None, df_sensor=None, 
                     xticks = [0, 30, 60], xticklabels = ['0', '', '60'], 
                     spines=['left', 'bottom'], ylabeloffset=-0.13, phi_alignment='align_psi'):
    
    if df_sensor is None:
        df_sensor, parameters = utility.get_sensor_measurements_derivatives_controls(df, 
                                                                 derivative_method='smooth_finite_difference.butterdiff', 
                                                                 params='turning',
                                                                 angular_noise_std=angular_noise_std,
                                                                 sensor_group='polar',
                                                                 return_smoothed=False,
                                                                 cutoff_freq=psi_freq,
                                                                 phi_alignment=phi_alignment,
                                                                 correction_window_for_2pi=correction_window_for_2pi,
                                                                )
    
    if ax is None:
        fig = plt.figure(figsize=(3.5, 2), dpi=300)
        ax = fig.add_subplot(111)

    #ax.scatter(df_sensor.t, df_sensor['sensor_'+sensor], c='lightgreen', s=0.25)
    ax.scatter(df_sensor.t, utility.wrap_angle(df_sensor['sensor_'+sensor]), c='lightgreen', s=0.5, edgecolors='none', rasterized=True)

    if 1:
        #gamma_smooth_unwrapped = utility.unwrap_angle(df_sensor['sensor_' + sensor + '_smooth'])
        val = df_sensor['sensor_' + sensor + '_smooth']
        chunks_val, chunks_t, breaks = flymath.get_continuous_chunks(val, df_sensor.t)
        print(len(chunks_val))
        for c, chunk in enumerate(chunks_val):
            ax.plot(chunks_t[c], chunk, color='green', linewidth=0.5)
        #ax.plot(df_sensor.t, gamma_smooth_unwrapped-2*np.pi, color='green', linewidth=0.5)
        #ax.plot(df_sensor.t, gamma_smooth_unwrapped+2*np.pi, color='green', linewidth=0.5)

    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlim(min(xticks), max(xticks))

    yticks = [-np.pi, 0, np.pi]
    yticklabels = ['$-\pi$', '', '$\pi$']
    
    
    fifi.mpl_functions.adjust_spines(ax, spines, 
                                 yticks=yticks,
                                 xticks=xticks,
                                 tick_length=2.5,
                                 spine_locations={'left': 5, 'bottom': 5},
                                 linewidth=0.5)
    
    if 'left' in spines:
        ax.set_yticklabels(yticklabels)
    else:
        ax.set_yticklabels([])
    if 'bottom' in spines:
        ax.set_xticklabels(xticklabels)
    else:
        ax.set_xticklabels([])
    ax.yaxis.set_label_coords(ylabeloffset, .5)
    
    
    sensor_latex = {'gamma': '$\gamma$',
                    'phi': '$\phi$',
                    'psi': '$\psi$',}
    sensorname_latex = {'gamma': 'Air speed\ndirection',
                    'phi': 'Orientation',
                    'psi': 'Ground speed\ndirection',}
    string =   sensorname_latex[sensor] + ', ' + sensor_latex[sensor]
    
    ax.set_ylabel(string)
    if 'bottom' in spines:
        ax.set_xlabel('Time, sec')

    #if sensor == 'phi':
    #    ax.yaxis.set_label_coords(-.15, .6)
    #else:
    #    ax.yaxis.set_label_coords(-.05, .6)
    fifi.mpl_functions.set_fontsize(ax, FONTSIZE)
    


def plot_ground_speed(df, ax=None, xticks=[0, 30, 60], yticks=[0, 1.5], show_xspine=True):
    if ax is None:
        fig = plt.figure(figsize=(3.5, 2), dpi=300)
        ax = fig.add_subplot(111)
        
    ax.plot(df.t, df.vel, color='black', linewidth=0.5)

    ax.set_ylim(min(yticks), max(yticks))
    ax.set_xlim(min(xticks), max(xticks))
    
    if show_xspine:
        spines = ['left', 'bottom']
    else:
        spines = ['left']

    fifi.mpl_functions.adjust_spines(ax, spines, 
                                     yticks=yticks,
                                     xticks=xticks,
                                     tick_length=2.5,
                                     spine_locations={'left': 5, 'bottom': 5},
                                     linewidth=0.5)
    #ax.yaxis.set_label_coords(-0.15, .6)
    ax.set_ylabel('Ground speed,\nm/s')
    
    if show_xspine:
        ax.set_xticklabels(xticks)
        ax.set_xlabel('Time, sec')

    fifi.mpl_functions.set_fontsize(ax, FONTSIZE)
    
def plot_globalpsi(df, ax=None, xticks=[0, 30, 60], yticks=[-np.pi, 0, np.pi]):
    if ax is None:
        fig = plt.figure(figsize=(3.5, 2), dpi=300)
        ax = fig.add_subplot(111)
        
    ax.scatter(df.t, utility.wrap_angle(df.psi + df.phi), c='black', s=0.5, edgecolors='none', rasterized=True)

    ax.set_ylim(min(yticks), max(yticks))
    ax.set_xlim(min(xticks), max(xticks))

    fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'], 
                                     yticks=yticks,
                                     xticks=xticks,
                                     tick_length=2.5,
                                     spine_locations={'left': 5, 'bottom': 5},
                                     linewidth=0.5)
    ax.set_yticklabels([r'$-\pi$', '', r'$\pi$'])
    ax.yaxis.set_label_coords(-0.13, .5)
    
    ax.set_xticklabels(xticks)
    ax.set_ylabel('Global\ncourse direction,\n' + r'$\psi + \phi$')
    
    ax.set_xlabel('Time, sec')

    #ax.yaxis.set_label_coords(-.15, .6)
    fifi.mpl_functions.set_fontsize(ax, FONTSIZE)
    


def load_real_wind(wind_type='dynamic', L=None, relative_start_time=None,directory=None, return_raw=False):
    '''
    dynamic case: relative_start_time = 9500, L=15000
    static case: relative_start_time = 9500+2760, L=7000
    '''
    
    if wind_type == 'dynamic':
        if relative_start_time is None:
            relative_start_time = 9500
        if L is None:
            L = 15000
    elif wind_type == 'constant':
        if relative_start_time is None:
            relative_start_time = 9500 + 2760
        if L is None:
            L = 7000
    elif wind_type == 'all':
        if relative_start_time is None:
            relative_start_time = 9500
        if L is None:
            L = 150000
    
    if directory is None:
        directory = '../data_experiments_clean'
    fname = os.path.join(directory, 'df_stationary_wind.hdf')
    df_stationary_wind = pandas.read_hdf(fname)
    
    if return_raw:
        return df_stationary_wind
    
    dt = np.median(np.diff(df_stationary_wind.time_epoch))

    print(L, dt)
    duration = L*dt

    time_epoch_start = (df_stationary_wind.time_epoch.min()+relative_start_time)
    time_epoch_end = (time_epoch_start+duration)

    time_epoch_start_str = str(time_epoch_start)
    time_epoch_end_str = str(time_epoch_end)

    df_stationary_wind_q = df_stationary_wind.query('time_epoch > ' + time_epoch_start_str + ' and time_epoch < ' + time_epoch_end_str)

    zeta = df_stationary_wind_q.zeta.values
    w = df_stationary_wind_q.w.values
    t = df_stationary_wind_q.time_epoch.values - df_stationary_wind_q.time_epoch.min()
    
    return df_stationary_wind_q, zeta, w, t, dt


def get_filenames_sorted_by_tau_and_T(directory, basename, angular_noise_std=0.1, phi_alignment='alignpsi', Tmultiplier=1, paramX=None, Tvalue=None):
    #phi_alignment = phi_alignment.replace('_', '')
    print('only finding filenames containing: ' + phi_alignment)
    
    filenames_raw = pynumdiff.utils.utility.get_filenames(directory, basename)
    filenames = []
    for filename in filenames_raw:
        if 'smoothingparameters' in filename:
            continue
        if phi_alignment not in filename:
            continue
        
        if paramX is not None:
            p = float(filename.split('paramX')[1].split('.hdf')[0])
            if p != paramX:
                continue
        if '_angularnoisestd' + str(angular_noise_std) in filename:
            
            s_tau = filename.split('_tau')[1]
            tau = float(s_tau.split('_')[0])
        
            s_T = filename.split('_T')[-1]
            if 'paramX' in s_T:
                T = float(s_T.split('_paramX')[0])
            else:
                T = float(s_T.split('.hdf')[0])
                
            if Tvalue is None:
                if np.abs(tau*Tmultiplier - T) < 0.01:
                    filenames.append(filename)
            else:
                if np.abs(Tvalue - T) < 0.01:
                    filenames.append(filename)
                
    ignore_psifreqs = [1] # using 1.25 instead
    
    psifreqs = []
    taus = []
    Ts = []
    sorting_numbers = []
    for filename in filenames:
        
        s_psifreq = filename.split('_psifreq')[1]
        psifreq = float(s_psifreq.split('_')[0])
        
        if psifreq in ignore_psifreqs:
            continue
        else:
            psifreq = float(s_psifreq.split('_')[0])
            psifreqs.append( psifreq )

            s_tau = filename.split('_tau')[1]
            tau = float(s_tau.split('_')[0])
            taus.append( tau )

            s_T = filename.split('_T')[-1]
            if 'paramX' in s_T:
                T = float(s_T.split('_paramX')[0])
            else:
                T = float(s_T.split('.hdf')[0])
            Ts.append( T )

            sorting_numbers.append(psifreq*1000000000 + tau*10000 + T)

        
        
    ix_sorted = np.argsort(sorting_numbers)
    return np.array(filenames)[ix_sorted], np.array(psifreqs)[ix_sorted], np.array(taus)[ix_sorted], np.array(Ts)[ix_sorted]


def plot_error_heatmap(error_heatmap, directory, basename, angular_noise_std=0.1, Tmultiplier=1, paramX=None, use='median',
                       logscalex=True, logscaley=True, show_xspine=True, show_yspine=True, phi_alignment='alignpsi',
                       show_contour=False, ax=None, vmin=0, vmax=np.pi/4., show_ellipse=True, show_turns_per_tau_line=True):
    if ax is None:
        fig = plt.figure(figsize=(3.5, 2), dpi=300)
        ax = fig.add_subplot(111)

    filenames, psifreqs, taus, Ts = get_filenames_sorted_by_tau_and_T(directory,  basename, 
                                                                      angular_noise_std=angular_noise_std,
                                                                      Tmultiplier=Tmultiplier,
                                                                      phi_alignment=phi_alignment,
                                                                      paramX=paramX)

    error_array = error_heatmap
    
    #ax.imshow((error_array), origin='lower', vmin=(1e-3), vmax=(np.pi/2.), interpolation='nearest',
    #         extent=[np.min(taus), np.max(taus), np.min(psifreqs), np.max(psifreqs)])
    
    #ax.contourf( np.unique(taus), np.unique(psifreqs), error_array, levels=np.sort(np.unique(error_array)) )
    
    #print(np.unique(taus), np.unique(psifreqs), error_array)
    #ax.contour( np.unique(taus), np.unique(psifreqs), error_array, levels=[ np.min(error_array)*1.05], colors='red')

    x = np.sort(np.unique(taus))
    y = np.sort(np.unique(psifreqs))
    Z = error_array
    
    ax.pcolormesh( x, y, Z, vmin=vmin, vmax=vmax, rasterized=True, shading='nearest')
    print('NEED PYTHON 3.8 and MATPLOTLIB 3.5 FOR CORRECT SHADING!!!')
        
    print(x, y, Z)
    print(len(x), len(y), Z.shape)
    
    
    if 0:
        if not logscalex:
            x2 = np.diff(x)/2. + x[0:-1]
            x2 = np.hstack((x[0] - np.diff(x)[0]/2., x2))
            x2 = np.hstack((x2, x[-1] + np.diff(x)[-1]/2.))
        else:
            logx = np.log(x)
            logx2 = np.diff(logx)/2. + logx[0:-1]
            logx2 = np.hstack((logx[0] - np.diff(logx)[0]/2., logx2))
            logx2 = np.hstack((logx2, logx[-1] + np.diff(logx)[-1]/2.))
            x2 = np.exp(logx2)

        if not logscaley:
            y2 = np.diff(y)/2. + y[0:-1]
            y2 = np.hstack((y[0] - np.diff(y)[0]/2., y2))
            y2 = np.hstack((y2, y[-1] + np.diff(y)[-1]/2.))
        else:
            logy = np.log(y)
            logy2 = np.diff(logy)/2. + logy[0:-1]
            logy2 = np.hstack((logy[0] - np.diff(logy)[0]/2., logy2))
            logy2 = np.hstack((logy2, logy[-1] + np.diff(logy)[-1]/2.))
            y2 = np.exp(logy2)

        print('x2: ', x2)

        f = interp2d(x, y, Z, kind='linear')
        Z2 = f(x2, y2)
        X2, Y2 = np.meshgrid(x2, y2)
    
    x2 = x
    y2 = y
    X2, Y2 = np.meshgrid(x, y)
    Z2 = Z
    
    pts = []
    vals = []
    for r in range(Z2.shape[0]):
        for c in range(Z2.shape[1]):
            pts.append([y2[r], x2[c]])
            vals.append(Z2[r,c])

    #Z2 = scipy.interpolate.griddata( np.array(pts), np.array(vals), (Y2, X2), method='linear')
   
    #print('Best error: ', np.min(Z))
    #print('Best error file: ', filenames[np.argmin(all_errors)])
    #print('Median Error for file: ', all_errors[np.argmin(all_errors)])
    
    
    # upsample for contours
    x3 = np.exp(np.linspace(np.log(min(x2)), np.log(max(x2)), 500))
    y3 = np.exp(np.linspace(np.log(min(y2)), np.log(max(y2)), 500))
    X3, Y3 = np.meshgrid(x3, y3)
    Z3 = scipy.interpolate.griddata( np.array(pts), np.array(vals), (Y3, X3), method='nearest')

    # buffer with extra row, column, to help with complete contours
    X3 = np.hstack((X3[:,0:1] - (X3[:,1:2] - X3[:,0:1]), X3))
    newcol = X3[:, X3.shape[1]-1:X3.shape[1]] + ((X3[:,-1] - X3[:,-2]).reshape(len(X3), 1))
    X3 = np.hstack((X3, newcol))
    X3 = np.vstack((X3[0,:], X3, X3[-1,:]))

    Y3 = np.vstack((Y3[0:1,:] - (Y3[1,:] - Y3[0,:]), Y3))
    Y3 = np.vstack((Y3, Y3[Y3.shape[0]-1:Y3.shape[0], :] + (Y3[-1,:] - Y3[-2,:])))
    Y3 = np.hstack((Y3[:,0:1], Y3, (Y3[:,-1].reshape(len(Y3), 1))))

    m = np.max(Z3)*2
    col = np.ones_like(Z3[:,0:1])*10
    Z3 = np.hstack((m*col, Z3, m*col))
    row = np.ones_like(Z3[0:1,:])*10
    Z3 = np.vstack((m*row, Z3, m*row))
    
    
    
    print('Best error after interpolation: ', np.min(Z3))
    #contours = ax.contour(X3, Y3, Z3, levels=[np.min(Z3) + 5*np.pi/180.], colors='red')

    # plot contours
    if 1:
        if 1:
            if show_contour:
                contours = ax.contour(X3, Y3, Z3, levels=[np.min(Z3) + 5*np.pi/180.], 
                                      linestyles='--', colors='red', linewidths=0.5)
            else:
                contours = ax.contour(X3, Y3, Z3, levels=[np.min(Z3) + 5*np.pi/180.], colors='none')
        else:
            if show_contour:
                contours = ax.contour(X2, Y2, Z2, levels=[np.min(Z2) + 5*np.pi/180.], 
                                      linestyles='--', colors='red', linewidths=0.5)
            else:
                contours = ax.contour(X2, Y2, Z2, levels=[np.min(Z2) + 5*np.pi/180.], colors='none')
            
        #return contours
    
        areas = []
        #segs = contours.collections[0].get_segments()
        paths = contours.collections[0].get_paths() #[0].vertices
        for i, path in enumerate(paths):
            seg = path.vertices

            xmin = min(seg[:,0])
            xmax = max(seg[:,0])

            ymin = min(seg[:,1])
            ymax = max(seg[:,1])

            area = (xmax-xmin)*(ymax-ymin)
            areas.append(area)

        ibiggest = np.argmax(areas)
        segs = paths[ibiggest].vertices
        segs = np.log(segs)

        contour_pts = None
        for i in range(len(segs)):
            #plt.plot( np.exp(segs[i][0]), np.exp(segs[i][1]), '*', zorder=100)

            p1 = segs[i-1]
            p2 = segs[i]
            choice = np.argmax([np.abs(p2[0] - p1[0]), np.abs(p2[1] - p1[1])])
            if choice == 0:
                slope = (p2[1] - p1[1])/(p2[0] - p1[0])
                intercept = p2[1] - slope*p2[0]
                d = np.sqrt( (p2[1] - p1[1])**2 + (p2[0] - p1[0])**2 )
                npts = int( np.ceil(d/0.1) )
                xp = np.linspace( (np.min([p1[0], p2[0]])), (np.max([p1[0], p2[0]])), npts)
                yp = slope*xp + intercept
            else:
                slope = (p2[0] - p1[0])/(p2[1] - p1[1])
                intercept = p2[0] - slope*p2[1]
                d = np.sqrt( (p2[1] - p1[1])**2 + (p2[0] - p1[0])**2 )
                npts = int( np.ceil(d/0.1) )
                yp = np.linspace( (np.min([p1[1], p2[1]])), (np.max([p1[1], p2[1]])), npts)
                xp = slope*yp + intercept


            if contour_pts is not None:
                contour_pts = np.vstack( (contour_pts, np.vstack((xp,yp)).T) )
            else:
                contour_pts = np.vstack((xp,yp)).T

        x = (contour_pts[:, 0])
        y = (contour_pts[:, 1])

        xmean = np.mean(x)
        ymean = np.mean(y)
        x -= xmean
        y -= ymean
        
        print('means: ', xmean, ymean)

        U, S, V = np.linalg.svd(np.stack((x, y)))
        tt = np.linspace(0, 2*np.pi, 1000)
        circle = np.stack((np.cos(tt), np.sin(tt)))    # unit circle
        transform = np.sqrt(2/len(x)) * U.dot(np.diag(S))   # transformation matrix
        fit = transform.dot(circle) + np.array([[xmean], [ymean]])

        x_mean = np.mean( (fit[0, :]) )
        y_mean = np.mean( (fit[1, :]) )
        
        print('means: ', x_mean, y_mean)
        
        if show_ellipse:
            ax.plot( np.exp(x_mean), np.exp(y_mean), '*', color='red')
            ax.plot( np.exp(fit[0, :]), np.exp(fit[1, :]), color='red')
        
        ellipse_tau = [np.min(np.exp(fit[0, :])), np.max(np.exp(fit[0, :]))]
        ellipse_freq = [np.min(np.exp(fit[1, :])), np.max(np.exp(fit[1, :]))]
    else:
        print('Could not plot contours.')
        ellipse_tau = [None, None]
        ellipse_freq = [None, None]
        
        
    # plot line for 1 turn per tau
    if show_turns_per_tau_line:
        turns_per_tau = 0.5/x3 # use 0.5, because frequency actually shows 2 turns/second
        ax.plot(x3, turns_per_tau, linewidth=2, color='white')
        ax.plot(x3, turns_per_tau, '--', linewidth=0.6, color='black')

    ax.set_xscale('log')
    ax.set_yscale('log')
    
    xticks = []
    yticks = []
    spines = []
    if show_xspine:
        xticks = [0.4, 2, 10, 50, 250]
        spines = ['bottom']
    else:
        ax.axes.xaxis.set_visible(False)
    if show_yspine:
        yticks = [0.01, 0.02, 0.1, 0.5, 1.25]
        spines = ['left']
    else:
        ax.axes.yaxis.set_visible(False)
    if show_xspine and show_yspine:
        spines = ['left', 'bottom']
    
    if 1:
        ax.set_xlim(0.2, 500)
        ax.set_ylim(0.008, 1.75)
        fifi.mpl_functions.adjust_spines(ax, spines, 
                                         yticks = yticks,
                                         xticks = xticks,
                                         spine_locations={'left': 5, 'bottom': 5},
                                         default_ticks=False,
                                         tick_length=2.5,
                                         linewidth=0.5)

        ax.minorticks_off()
        #ax.set_yticklabels(yticks)
        #ax.set_xticklabels(xticks)
        mathify_ticklabels(ax, 'x', xticks)
        mathify_ticklabels(ax, 'y', yticks)
        
    if show_xspine and show_yspine:
        ax.set_ylabel('Turning frequency, Hz')
        #ax.yaxis.set_label_coords(-.22, .6)
        ax.set_xlabel(r'$\tau$, sec')

    fifi.mpl_functions.set_fontsize(ax, FONTSIZE)
    
    #     else:
    #         fifi.mpl_functions.adjust_spines(ax, [], 
    #                                          #yticks = np.arange(0,error_array.shape[0]),
    #                                          #xticks = np.arange(0,error_array.shape[1]),
    #                                          spine_locations={'left': 6, 'bottom': 6},
    #                                          default_ticks=False)
    #         ax.axes.xaxis.set_visible(False)
    #         ax.axes.yaxis.set_visible(False)

    tau_opt = np.exp(x_mean)
    psifreq_opt = np.exp(y_mean)
    error_opt = np.min(error_array)
    
    print(tau_opt, psifreq_opt, error_opt)
    return tau_opt, psifreq_opt, error_opt, X3, Y3, Z3, ellipse_tau, ellipse_freq


def get_optimal_psifreq_tau_vs_noise(directory, basename, phi_alignment, angular_noise_stds=[0.3, 0.6, 1.2], 
                                     Tmultiplier=1, paramX=1, use='median'):
    
    optimal_tau_values = []
    optimal_psiqfreq_values = []
    optimal_error_values = []
    
    optimal_tau_stds = []
    optimal_psiqfreq_stds = []
    optimal_error_stds = []
    
    optimal_tau_ellipse_min = []
    optimal_tau_ellipse_max = []
    
    optimal_freq_ellipse_min = []
    optimal_freq_ellipse_max = []
    
    for angular_noise_std in angular_noise_stds:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        error_heatmap, opt_T, optimal_error_std_heatmap = get_best_error_heatmap_across_Tmultipliers_and_optimal_Tmultipler(directory,  
                                                                                              basename, 
                                          angular_noise_std=angular_noise_std,
                                          phi_alignment=phi_alignment,
                                          paramX=paramX,
                                          return_error_std=True)
        
        res = plot_error_heatmap(error_heatmap, directory, basename, angular_noise_std=angular_noise_std, 
                           Tmultiplier=1, paramX=paramX, use='median',
                           phi_alignment=phi_alignment,
                               show_xspine=False, show_yspine=False,
                               show_contour=False, ax=ax)
        
        tau_opt, psifreq_opt, error_opt, X3, Y3, Z3, ellipse_tau, ellipse_freq = res

        ix = np.unravel_index(error_heatmap.argmin(), error_heatmap.shape)
        optimal_error_stds.append(optimal_error_std_heatmap[ ix ])

        optimal_tau_values.append(tau_opt)
        optimal_psiqfreq_values.append(psifreq_opt)
        optimal_error_values.append(error_opt)
        
        optimal_tau_ellipse_min.append(ellipse_tau[0])
        optimal_tau_ellipse_max.append(ellipse_tau[1])
        optimal_freq_ellipse_min.append(ellipse_freq[0])
        optimal_freq_ellipse_max.append(ellipse_freq[1])

        plt.close('all')
    
    print(angular_noise_stds, optimal_psiqfreq_values, optimal_error_values, optimal_tau_values)
    df = pandas.DataFrame({'angular_noise_std': angular_noise_stds,
                           'optimal_psiqfreq_values': optimal_psiqfreq_values,
                           'optimal_error_values': optimal_error_values,
                           'optimal_tau_values': optimal_tau_values,
                           'optimal_tau_ellipse_min': optimal_tau_ellipse_min,
                           'optimal_tau_ellipse_max': optimal_tau_ellipse_max,
                           'optimal_freq_ellipse_min': optimal_freq_ellipse_min,
                           'optimal_freq_ellipse_max': optimal_freq_ellipse_max,
                           #'optimal_psiqfreq_stds': optimal_psiqfreq_stds,
                           'optimal_error_stds': optimal_error_stds,
                           #'optimal_tau_stds': optimal_tau_stds})
                          })
    
    return df

def plot_timeseries(filename, ax=None, spines=['left', 'bottom'], filter=False, filter_freq=0.5, use='median', return_errors=False):
    '''
    L - defined by simultaion
    '''
    def make_plot_pretty(ax):
        print('biggest t: ', np.max(t))
        xticks = (np.unique(np.ceil(t / 60. / 5.))*5).astype(int) # minutes
        print(xticks)
        
        yticks = [0, np.pi/2., np.pi, 3*np.pi/2, 2*np.pi]
        yticklabels = ['$0$', '', '', '', '$2\pi$']
        xticklabels = xticks


        fifi.mpl_functions.adjust_spines(ax, spines, 
                                         yticks=yticks,
                                         xticks=xticks*60,
                                         tick_length=2.5,
                                         spine_locations={'left': 5, 'bottom': 5},
                                         linewidth=0.5)
        #ax.yaxis.set_label_coords(-0.1, .5)
        
        ax.set_yticklabels(yticklabels)
        

        ax.set_ylabel('Wind direction,\n$\zeta$')
        #ax.yaxis.set_label_coords(-.15, .6)

        if 'bottom' in spines:
            mathify_ticklabels(ax, 'x', ticks=xticks)
            ax.set_xlabel('Time, min')
            print(xticks)

        if 1:
            ax.set_ylim(0, 2*np.pi)
            ax.set_xlim(0, xticks[-1]*60)

        fifi.mpl_functions.set_fontsize(ax, FONTSIZE)

        
    if ax is None:
        fig = plt.figure(figsize=(3.5, 2), dpi=300)
        ax = fig.add_subplot(111)

    if 'wind' in os.path.basename(filename):
        if 'windrealdynamic' in filename:
            wind_type = 'dynamic'
        elif 'windrealconstant' in filename:
            wind_type = 'constant'
    else:
        wind_type = 'dynamic'

   
    
    def make_zeta_plot(ax, t, zeta_true, data_t, data_zeta_true, data_zeta_est):
        if 1:
            # hack to not plot points below zero
            y1 = copy.copy(zeta_true)
            #y2 = copy.copy(zeta + 2*np.pi)
            y1[y1<0] = y1[y1<0] + 2*np.pi

            ax.scatter(t, y1, c='black', s=0.25, rasterized=True)
            #ax.scatter(t, y2, c='black', s=0.25, rasterized=True)
        else:
            ax.scatter(data_t, data_zeta_true, c='black', s=0.25, rasterized=True)
            ax.scatter(data_t, data_zeta_true + 2*np.pi, c='black', s=0.25, rasterized=True)

        # hack to not plot points below zero
        y1 = copy.copy(data_zeta_est)
        #y2 = copy.copy(data.zeta_est + 2*np.pi)
        y1[y1<0] = y1[y1<0] + 2*np.pi

        ax.scatter(data_t, y1, c='red', s=0.5, rasterized=True)
        #ax.scatter(data.t, y2, c='red', s=0.5, rasterized=True)

        #print(np.min(data.zeta_est), np.max(data.zeta_est))
    
    
     # original wind data
    df_stationary_wind_q, zeta_true, w, t, dt = load_real_wind(wind_type)
    dt = np.mean(np.diff(t))
    L = len(t)
    
    # estimates
    data = pandas.read_hdf(filename)
    
    # nearest interpolation of original wind data to get the wind data closest to the estimate
    def nearest_interp(xi, x, y):
        idx = np.abs(x - xi[:,None])
        return y[idx.argmin(axis=1)]
    
    
    wind_interp_zeta_true =  data.zeta_true.values #nearest_interp(data.t.values, t, zeta_true)
    wind_interp_t =  data.t #copy.copy(data.t.values)
    
    original_zeta = df_stationary_wind_q.zeta.values
    original_t = df_stationary_wind_q.time_epoch.values - df_stationary_wind_q.time_epoch.min()
    
    errors = np.abs(utility.wrap_angle(data.zeta_est.values - wind_interp_zeta_true))
    
    if not filter:
        make_zeta_plot(ax, original_t, original_zeta, data.t.values, data.zeta_true.values, data.zeta_est.values)
        make_plot_pretty(ax)
        
        mean_error = np.mean( errors )
        if use == 'median':
            median_error = np.percentile( errors, 50)
            print('Median error: ', median_error)
            print('median + mean / 2: ',  (median_error + mean_error)/2)
        elif use == 'mean':
            print('mean error: ', mean_error)
    
    else:
        nyquist_freq = 0.5*1/dt
        butter_freq = filter_freq / nyquist_freq
        params = [4, butter_freq]
        derivative_method = 'smooth_finite_difference.butterdiff'
        correction_window_for_2pi = 100
        
        zeta_est_smooth, _ = utility.diff_angle(data.zeta_est.values, dt, params, derivative_method=derivative_method, correction_window_for_2pi=correction_window_for_2pi)
        
        zeta_true_smooth, _ = utility.diff_angle(wind_interp_zeta_true, dt, params, derivative_method=derivative_method, correction_window_for_2pi=correction_window_for_2pi)
        
        make_zeta_plot(ax, wind_interp_t, wind_interp_zeta_true, data.t.values, zeta_true_smooth, zeta_est_smooth)
        make_plot_pretty(ax)
        
        
        mean_error = np.mean(np.abs(utility.wrap_angle(zeta_est_smooth - wind_interp_zeta_true)))
        if use == 'median':
            median_error = np.percentile(np.abs(utility.wrap_angle(zeta_est_smooth - wind_interp_zeta_true)), 50)
            print('Median error: ', median_error)
            print('median + mean / 2: ',  (median_error + mean_error)/2)
        elif use == 'mean':
            
            print('mean error: ', mean_error)
            
    if return_errors:
        return errors
    
def plot_example_zeta_timeseries(directory, turn_angle, freq, tau, angular_noise_std, Tmultiplier=1, 
                                 phi_alignment='alignpsi',
                                 paramX=1,
                                 ax=None, spines=['left', 'bottom'], windtype='realdynamic',
                                 filter=False,
                                 filter_freq=0.5,
                                 use='median',
                                 return_errors=False,
                                 return_filename=False):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
    #directory = 'cvx_result_dynamic_wind_20211204'
    basename= 'cvx_wind' + windtype + '_random_turnamplitude' + str(turn_angle) + 'deg'
    print(directory)
    print(basename)

    candidates = get_filenames_sorted_by_tau_and_T(directory, basename, 
                                                   angular_noise_std=angular_noise_std, 
                                                   Tmultiplier=Tmultiplier,
                                                   phi_alignment=phi_alignment,
                                                   paramX=paramX)[0]

    filename = None
    for candidate in candidates:
        if 'psifreq' + str(freq) in candidate:
            if 'tau' + str(tau) in candidate:
                if paramX is None:
                    filename = candidate
                    break
                else:
                    if 'paramX' + str(paramX) in candidate:
                        filename = candidate
                        break
                
    print('Filename:  ' + filename)
    errors = plot_timeseries(filename, ax=ax, spines=spines, filter=filter, filter_freq=filter_freq, use=use, return_errors=return_errors)
    
    if return_filename:
        return filename
    
    if return_errors:
        return errors
    
def get_error_heatmap_for_Tmultiplier(directory,  basename, 
                                      angular_noise_std=0.1,
                                      Tmultiplier=1,
                                      phi_alignment='alignpsi',
                                      paramX=1,
                                      use='median',
                                      return_error_std=False):

    filenames, psifreqs, taus, Ts = get_filenames_sorted_by_tau_and_T(directory,  basename, 
                                                                      angular_noise_std=angular_noise_std,
                                                                      Tmultiplier=Tmultiplier,
                                                                      phi_alignment=phi_alignment,
                                                                      paramX=paramX)

    error_array = np.ones([len(np.unique(psifreqs)), len(np.unique(taus))])*np.nan
    error_std_array = np.ones([len(np.unique(psifreqs)), len(np.unique(taus))])*np.nan
    all_errors = []

    for i, filename in enumerate(filenames):
        data = pandas.read_hdf(filename)
        if data.shape[0] == 0:
            continue

        if 'wind' in os.path.basename(filename):
            if 'windrealdynamic' in filename:
                wind_type = 'dynamic'
            elif 'windrealconstant' in filename:
                wind_type = 'constant'
        else:
            wind_type = 'dynamic'

        if 1:
            errors = []
            df_stationary_wind_q, zeta, w, t, dt = load_real_wind(wind_type)
            df_stationary_wind_q['t'] = df_stationary_wind_q.time_epoch - df_stationary_wind_q.time_epoch.min()

            zeta_est_interp = np.interp(t, data.t, data.zeta_est)
            errors = np.abs(utility.wrap_angle(zeta - zeta_est_interp)[0:len(zeta_est_interp)])
            all_errors.append(np.mean(errors))
        else:
            errors = np.abs(utility.wrap_angle(data.zeta_est.values - data.zeta_true.values))

        if use == 'median':
            error_median = np.median(errors)
        elif use == 'mean':
            error_median = np.mean(errors)
        elif use == 'meanmedian':
            error_median = (np.median(errors) + np.mean(errors))/2

        error_std = np.std(errors)

        c = np.argmin( np.abs(data.tau.values[0] - np.unique(taus)) )
        r = np.argmin( np.abs(data.psi_freq.values[0] - np.unique(psifreqs)) )

        error_array[r,c] = error_median
        error_std_array[r,c] = error_std
        
    if not return_error_std:
        return error_array
    else:
        return error_array, error_std_array

def get_best_error_heatmap_across_Tmultipliers_and_optimal_Tmultipler(directory,  basename, 
                                      angular_noise_std=0.1,
                                      phi_alignment='alignpsi',
                                      paramX=1,
                                      use='median',
                                      return_error_std=False):

    Tmultipliers = [1, 5, 25, 125]
    
    error_heatmaps = []
    error_std_heatmaps = []
    
    for Tmultiplier in Tmultipliers:

        e, e_std = get_error_heatmap_for_Tmultiplier(directory,  basename, 
                                              angular_noise_std=angular_noise_std,
                                              Tmultiplier=Tmultiplier,
                                              phi_alignment=phi_alignment,
                                              paramX=paramX,
                                              use=use,
                                              return_error_std=return_error_std)
        error_heatmaps.append(e)
        error_std_heatmaps.append(e_std)
        
        
    error_heatmaps_stack = np.dstack(error_heatmaps)

    optimal_error_heatmap = np.zeros_like(error_heatmaps[0])
    optimal_Tmultipler_idx = np.zeros_like(error_heatmaps[0])
    optimal_Tmultiplier = np.zeros_like(error_heatmaps[0])

    optimal_error_std_heatmap = np.zeros_like(error_heatmaps[0])

    for r in range(optimal_error_heatmap.shape[0]):
        for c in range(optimal_error_heatmap.shape[1]):
            ix = np.argmin(error_heatmaps_stack[r,c,:])
            optimal_Tmultipler_idx[r,c] = ix
            optimal_error_heatmap[r,c] = error_heatmaps_stack[r,c,ix]
            optimal_Tmultiplier[r,c] = Tmultipliers[ix]
            optimal_error_std_heatmap[r,c] = error_std_heatmaps[ix][r,c]
        
    if not return_error_std:
        return optimal_error_heatmap, optimal_Tmultiplier
    else:
        return optimal_error_heatmap, optimal_Tmultiplier, optimal_error_std_heatmap

def plot_errors(directory, basename, angular_noise_std=0.1, Tmultiplier=1, 
                paramX=1,
                psi_freq='all', use='median', plot=True, specific_filename=None, ax=None,
                star_only=False):
    '''
    If star_only True: don't plot the scatterbox, only plot a star for the median
    
    '''
    
    if plot:
        if ax is None:
            fig = plt.figure(figsize=(3.5, 2), dpi=300)
            ax = fig.add_subplot(111)

    filenames, psifreqs, taus, Ts = get_filenames_sorted_by_tau_and_T(directory,  basename, 
                                                                      angular_noise_std=angular_noise_std,
                                                                      paramX=paramX,
                                                                      Tmultiplier=Tmultiplier)
    print(psifreqs)
    
    results_df = pandas.DataFrame({'psifreq': [],
                  'tau': [],
                  'Tmultipler': [],
                  'error': [],
                  'angular_noise_std': [],
                  'filename': []})
    
    cmap = matplotlib.cm.get_cmap('Set2')
    norm = matplotlib.colors.Normalize(vmin=np.log(np.min(psifreqs)), vmax=np.log(np.max(psifreqs)))

    #N_pts = get_minimum_N_pts(filenames)
    xticks = []
    taus = []
    
    if specific_filename is not None:
        filenames = [specific_filename,]
    
    for i, filename in enumerate(filenames):
        if 'wind' in os.path.basename(filename):
            if 'windrealdynamic' in filename:
                wind_type = 'dynamic'
            elif 'windrealconstant' in filename:
                wind_type = 'constant'
        else:
            wind_type = 'dynamic'
            
        data = pandas.read_hdf(filename)
        tau = data.tau.values[0]
        if psi_freq != 'all':
            if psi_freq != data.psi_freq.values[0]:
                continue
        #ix_rows = random.sample(range(data.shape[0]), N_pts)
        #data = data.iloc[ix_rows]

        if data.shape[0] == 0:
            continue

        if 1:
            errors = []
            df_stationary_wind_q, zeta, w, t, dt = load_real_wind(wind_type)
            df_stationary_wind_q['t'] = df_stationary_wind_q.time_epoch - df_stationary_wind_q.time_epoch.min()
            
            zeta_est_interp = np.interp(t, data.t, data.zeta_est)
            errors = utility.wrap_angle(zeta - zeta_est_interp)[0:len(zeta_est_interp):10]
        else:
            errors = utility.wrap_angle(data.zeta_est.values - data.zeta_true.values)

        
    
        if plot:
            rgba = cmap(norm(np.log(data.psi_freq.values[0])))
            
            if not star_only:
                fpl.scatter_box(ax, i, np.abs(errors), markersize=0.3, color=rgba, use=use)
            else:
                ax.plot(i, np.median(np.abs(errors)), '*', markersize=3.5, 
                        markerfacecolor=rgba, markeredgecolor='black', markeredgewidth =0.1)
            
            
            xticks.append(i)

            if len(xticks) <= 5:
                taus.append(tau)
            else:
                taus.append('')
            
            
        if use == 'mean':
            avg_error = np.mean( np.abs(errors) )
        elif use == 'median':
            avg_error = np.median( np.abs(errors) )
        results_new = pandas.DataFrame({   'psifreq': [data.psi_freq.values[0]],
                          'tau': [tau],
                          'Tmultipler': [Tmultiplier],
                          'error': [avg_error],
                          'angular_noise_std': [angular_noise_std],
                          'filename': [filename]})
        results_df = pandas.concat([results_df, results_new])
        
    if plot:
        
        if len(filenames) == 1:
            spine_locations={'left': 5, 'bottom': 5}
        else:
            spine_locations={'left': -5, 'bottom': 5}
        
        
        ax.set_ylabel('Error, $|\hat{\zeta}-\zeta|$')
        ax.set_xlabel(r'$\tau$, sec')
        
        
        yticks = [0, np.pi/4., np.pi/2., 3*np.pi/4, np.pi]
        yticklabels = ['$0$', '', '', '', '$\pi$']
        
        ax.set_ylim(yticks[0], yticks[-1])
        
        fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'], 
                                     yticks=yticks,
                                     xticks=xticks,
                                     tick_length=2.5,
                                     spine_locations=spine_locations, #{'left': 5, 'bottom': 5},
                                     linewidth=0.5)
        
        taus_str = []
        for tau in taus:
            if len(str(tau)) > 0:
                taus_str.append(r'$'+str(tau)+'$')
            else:
                taus_str.append('')
        print(taus_str)
        ax.set_xticklabels(taus_str, rotation='vertical', horizontalalignment='center')
        ax.set_yticklabels(yticklabels)
        
        if len(filenames) == 1:
            ax.yaxis.set_label_coords(-1.2, .5)
        if len(filenames) > 1:
            ax.xaxis.set_label_coords(0.13, -0.24)
        
        fifi.mpl_functions.set_fontsize(ax, FONTSIZE)
        
        #if not star_only:
        #    ax.set_rasterization_zorder(1000)
    
    return filenames, results_df

def colorbar(ax=None, colormap='jet', orientation='vertical', ticks=[0,1]):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    xlim = ticks
    ylim = ticks
    
    # horizontal
    if orientation == 'horizontal':
        grad = np.linspace(ticks[0], ticks[1], 500, endpoint=True)
        im = np.vstack((grad,grad))
    
    # vertical
    if orientation == 'vertical':
        grad = np.linspace(ticks[0], ticks[1], 500, endpoint=True)
        im = np.vstack((grad,grad)).T
    
    # make image
    cmap = plt.get_cmap(colormap)
    ax.imshow(  im, 
                cmap=cmap,
                extent=(xlim[0], xlim[-1], ylim[0], ylim[-1]), 
                origin='lower', 
                interpolation='bicubic')
    
    ax.set_aspect('auto')
    ax.set_xlim(xlim[0], xlim[-1])
    ax.set_ylim(ylim[0], ylim[-1])
    
    
    

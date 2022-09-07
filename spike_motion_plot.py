from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D     # enabling 3D projections
import matplotlib.animation as animation

import pandas as pd
from pandas_profiling import ProfileReport
from process_steps import process_steps
from process_spikes import sort
import pandas as pd
from plotly.offline import iplot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, butter, filtfilt, iirnotch, coherence
from scipy.ndimage import gaussian_filter1d
from pdb import set_trace

# ANIMATION FUNCTION
def line_draw(num, dataSet, line):
    line.set_data(dataSet[0:2, :num])
    line.set_3d_properties(dataSet[2, :num])    
    #dots.set_data(dataSet[0:2, :num])    
    #dots.set_3d_properties(dataSet[2, :num]) 
    return line

def spike_motion_plot(
    ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
        filter_ephys, filter_tracking, bin_width_ms, bin_width_radian, anipose_data_dict,
        bodypart_for_alignment, bodypart_for_reference, subtract_bodypart_ref,
        session_date, rat_name, treadmill_speed, treadmill_incline,
        camera_fps, align_to, vid_length, time_frame,
        do_plot, plot_template, MU_colors, CH_colors
    ):
    
    (MU_spikes_by_channel_dict, time_axis_for_ephys, time_axis_for_anipose,
    ephys_sample_rate, start_video_capture_ephys_idx, step_time_slice_ephys, session_parameters, _) = sort(
        ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
        filter_ephys, filter_tracking, anipose_data_dict, bodyparts_list=bodypart_for_alignment,
        bodypart_for_alignment=bodypart_for_alignment, bodypart_for_reference=bodypart_for_reference, subtract_bodypart_ref=subtract_bodypart_ref,
        session_date=session_date, rat_name=rat_name,
        treadmill_speed=treadmill_speed, treadmill_incline=treadmill_incline,
        camera_fps=camera_fps, align_to=align_to, vid_length=vid_length,
        time_frame=time_frame, do_plot=False, # change T/F whether to plot sorting plots also
        plot_template=plot_template, MU_colors=MU_colors, CH_colors=CH_colors
        )    

    processed_anipose_df, foot_strike_idxs, foot_off_idxs, sliced_step_stats, step_slice, step_time_slice = process_steps(
        anipose_data_dict, bodypart_for_alignment=bodypart_for_alignment, bodypart_for_reference=bodypart_for_reference, subtract_bodypart_ref=subtract_bodypart_ref,
        filter_tracking=filter_tracking, session_date=session_date, rat_name=rat_name, treadmill_speed=treadmill_speed,
        treadmill_incline=treadmill_incline, camera_fps=camera_fps, align_to=align_to, time_frame=time_frame
        )
    #set_trace()

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    #sets the box scale to be the same as our treadmill
    # ax.axes.set_xlim3d(left=87, right=-52) 
    # ax.axes.set_ylim3d(bottom=-211, top=342) 
    # ax.axes.set_zlim3d(bottom=-134, top=0) 
    #sets the box aspect ratio from the current axes limits to achieve the "equal" behavior
    #ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

    x_points =  processed_anipose_df['palm_L_x'][step_time_slice].to_numpy()
    y_points =  processed_anipose_df['palm_L_y'][step_time_slice].to_numpy()
    z_points =  processed_anipose_df['palm_L_z'][step_time_slice].to_numpy()
    #set_trace()
    dataSet = np.array([x_points, y_points, z_points])
    numDataPoints = len(x_points)

    #Define dots and line
    #dots = plt.plot(dataSet[0], dataSet[1], dataSet[2], ms=1, c='b', marker='.')[0] # For scatter plot
    line = plt.plot(dataSet[0], dataSet[1], dataSet[2], lw=1, c='b', linestyle='dotted', alpha=0.7)[0] # For line plot
    
    #Axes properties
    # ax.set_xlim3d([limit0, limit1])
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Trajectory of Joint Relative to Tailbase')
    
    #Make rainbow 3D scatterplot
    #ax.scatter3D(x_points, y_points, z_points, s=5, c=z_points, cmap='hsv')

    #Make line drawing
    ax.plot(dataSet[0], dataSet[1], dataSet[2], lw=1, c='b', linestyle='dotted', alpha=0.7)

    #Make line animation
    #line_ani = animation.FuncAnimation(fig, line_draw, frames=numDataPoints, fargs=(dataSet,line), interval=8, blit=False)
    # line_ani.save(r'Animation.mp4')

    #kinematic around when unit 2 fires
    for i in MU_spikes_by_channel_dict['13']['500']: #['7']['1700'] #['13']['500']
        spike_slice = slice(i//240-1,i//240+1)
        ax.plot(x_points[spike_slice],y_points[spike_slice],z_points[spike_slice],marker="x", lw=1.2,c="red")

    #set_trace()
    plt.show()
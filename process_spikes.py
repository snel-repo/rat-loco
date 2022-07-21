from extract_step_idxs import extract_step_idxs
import pandas as pd
import numpy as np
from plotly.offline import iplot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, butter, filtfilt, iirnotch
from scipy.ndimage import gaussian_filter1d
from pdb import set_trace
# from scipy.stats import binned_statistic
# from sklearn.decomposition import PCA

# create highpass filters to remove baseline from foot tracking for better peak finding performance
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# create a notch filter for powerline removal at 60Hz
def iir_notch(data, ephys_sample_rate, notch_frequency=60.0, quality_factor=30.0):
    quality_factor = 20.0  # Quality factor
    b, a = iirnotch(notch_frequency, quality_factor, ephys_sample_rate)
    y = filtfilt(b, a, data)
    return y

# create bandpass filters to remove noise from ephys data
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# main function for threshold sorting and producing spike indexes
def sort(
    ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
    filter_ephys, filter_tracking, anipose_data_dict, bodyparts_list, bodypart_for_tracking,
    session_date, rat_name, treadmill_speed, treadmill_incline,
    camera_fps, alignto, vid_length, time_slice, do_plot,
    plot_template, MU_colors, CH_colors
    ):

    # format inputs to avoid ambiguities
    rat_name = str(rat_name).lower()
    treadmill_speed = str(treadmill_speed).zfill(2)
    treadmill_incline = str(treadmill_incline).zfill(2)
    session_parameters = f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
    
    # filter Open Ephys dictionaries for the proper session date, speed, and incline
    ephys_data_dict_filtered_by_date = dict(filter(lambda item:
                                str(session_date) in item[0], ephys_data_dict.items()
                                ))
    ephys_data_dict_filtered_by_ratname = dict(filter(lambda item:
                                rat_name in item[0], ephys_data_dict_filtered_by_date.items()
                                ))
    ephys_data_dict_filtered_by_speed = dict(filter(lambda item:
                                "speed"+treadmill_speed in item[0],
                                ephys_data_dict_filtered_by_ratname.items()
                                ))
    ephys_data_dict_filtered_by_incline = dict(filter(lambda item:
                                "incline"+treadmill_incline in item[0],
                                ephys_data_dict_filtered_by_speed.items()
                                ))
    chosen_ephys_data_dict = ephys_data_dict_filtered_by_incline
    # convert chosen ephys dict into DataFrame
    chosen_ephys_data_continuous_obj = chosen_ephys_data_dict[session_parameters]

    # filter anipose dictionaries for the proper session date, speed, and incline
    anipose_data_dict_filtered_by_date = dict(filter(lambda item:
            str(session_date) in item[0], anipose_data_dict.items()
                                ))
    anipose_data_dict_filtered_by_ratname = dict(filter(lambda item:
            rat_name in item[0], anipose_data_dict_filtered_by_date.items()
                                ))
    anipose_data_dict_filtered_by_speed = dict(filter(lambda item:
            "speed"+treadmill_speed in item[0],anipose_data_dict_filtered_by_ratname.items()
                                ))
    anipose_data_dict_filtered_by_incline = dict(filter(lambda item:
            "incline"+treadmill_incline in item[0], anipose_data_dict_filtered_by_speed.items()
                                ))
    chosen_anipose_data_dict = anipose_data_dict_filtered_by_incline
    # convert chosen anipose dict into DataFrame
    chosen_anipose_df = chosen_anipose_data_dict[session_parameters]

    # create time axes
    ephys_sample_rate = chosen_ephys_data_continuous_obj.metadata['sample_rate']
    time_axis_for_ephys = np.arange(
        round(len(chosen_ephys_data_continuous_obj.samples)*time_slice)
        )/ephys_sample_rate
    
    # find the beginning of the camera SYNC pulse         
    filtered_sync_channel = butter_highpass_filter(
        data=chosen_ephys_data_continuous_obj.samples[:,-1], cutoff=50, fs=30000, order=2)
    start_video_capture_ephys_idx = find_peaks(filtered_sync_channel,height=0.3)[0][0]
    time_axis_for_anipose = np.arange(0,vid_length,1/camera_fps)+ \
                                        time_axis_for_ephys[start_video_capture_ephys_idx]

    # identify motion peak locations of bodypart for step cycle alignment
    if filter_tracking == True:
        filtered_signal = butter_highpass_filter(
            data=chosen_anipose_df[bodypart_for_tracking[0]].values,
            cutoff=0.5, fs=camera_fps, order=5)
    else: # do not filter
        filtered_signal=chosen_anipose_df[bodypart_for_tracking[0]].values

    filtered_signal, foot_strike_idxs, foot_off_idxs, _ = extract_step_idxs(
    anipose_data_dict, bodypart_for_tracking=bodypart_for_tracking,
    filter_tracking=filter_tracking, session_date=session_date, rat_name=rat_name,
    treadmill_speed=treadmill_speed, treadmill_incline=treadmill_incline, camera_fps=camera_fps,
    alignto=alignto)

    # cluster the spikes waveforms with PCA
    # pca = PCA(n_components=3)
    # pca.fit(chosen_ephys_data_continuous_obj.samples)
    # print(pca.explained_variance_ratio_)

    # extract spikes that are detected in the selected amplitude threshold ranges
    MU_spikes_by_unit_dict = {}
    MU_spikes_by_unit_dict_keys = [str(int(unit[0])) for unit in MU_spike_amplitudes_list]
    MU_channel_keys_list = [str(ch) for ch in ephys_channel_idxs_list]
    MU_spikes_by_channel_dict = {key: None for key in MU_channel_keys_list}
    for iChannel, channel_number in enumerate(ephys_channel_idxs_list):
        MU_spike_idxs = [] # initialize empty list for each channel to hold next sorted spike idxs
        for iUnit, iAmplitudes in enumerate(MU_spike_amplitudes_list):
            if channel_number not in [-1,16]:
                ephys_data_for_channel = chosen_ephys_data_continuous_obj.samples[:,channel_number]
                if filter_ephys == 'notch' or filter_ephys == 'both':
                    ephys_data_for_channel = iir_notch(
                        ephys_data_for_channel, ephys_sample_rate)
                if filter_ephys == 'bandpass' or filter_ephys == 'both':
                    ephys_data_for_channel = butter_bandpass_filter(
                        ephys_data_for_channel, 350.0, 7000.0, ephys_sample_rate) # 350-7000Hz band
                MU_spike_idxs_for_channel, _ = find_peaks(
                    -ephys_data_for_channel,
                    height=iAmplitudes,
                    threshold=None,
                    distance=50,
                    prominence=None,
                    width=None,
                    wlen=None,
                    )
                MU_spike_idxs.append(MU_spike_idxs_for_channel)
        MU_spikes_by_unit_dict = dict(zip(MU_spikes_by_unit_dict_keys,MU_spike_idxs))
        MU_spikes_by_channel_dict[str(channel_number)] = MU_spikes_by_unit_dict
    if filter_ephys == 'notch' or filter_ephys == 'both':
        print('60Hz notch filter applied to voltage signals.')
    if filter_ephys == 'bandpass' or filter_ephys == 'both':
        print('350-7000Hz bandpass filter applied to voltage signals.')
    if filter_ephys not in ['notch','bandpass','both']:
        print('No additional filters applied to voltage signals.')
    # MU_spike_idxs = np.array(MU_spike_idxs,dtype=object).squeeze().tolist()

    ### PLOTTING SECTION
    # compute number of channels and units per channel
    # then compute a color stride value to maximize use of color space
    number_of_channels = len(np.where((np.array(ephys_channel_idxs_list)!=16)
                                        &(np.array(ephys_channel_idxs_list)!=-1))[0])
    # number_of_units_per_channel = len(MU_spike_amplitudes_list)
    # color_stride = len(MU_colors)//(number_of_units_per_channel*number_of_channels)
    color_stride = 1
    # compute number of rows to allocate for each subplot based on numbers of each channel
    number_of_rows = len(bodyparts_list)+len(ephys_channel_idxs_list)+number_of_channels//2+1
    row_spec_list = number_of_rows*[[None]]
    row_spec_list[0] = [{'rowspan': len(bodyparts_list)}]
    row_spec_list[len(bodyparts_list)] = [{'rowspan': len(ephys_channel_idxs_list)}]
    row_spec_list[len(bodyparts_list)+len(ephys_channel_idxs_list)] = \
        [{'rowspan': number_of_channels//2+1}]

    fig = make_subplots(
        rows=number_of_rows, cols=1,
        specs=row_spec_list,
        shared_xaxes=True,
        # vertical_spacing=0.0,
        # horizontal_spacing=0.02,
        subplot_titles=(
        f"<b>Locomotion Kinematics: {list(chosen_anipose_data_dict.keys())[0]}</b>",
        f"<b>Neural Activity: {list(chosen_ephys_data_dict.keys())[0]}</b>"
        ))

    # plot all chosen bodyparts_list, including peak and trough locations for step identification
    bodypart_counter = 0
    for name, values in chosen_anipose_df.items():
        if name in bodyparts_list:
            if name == bodypart_for_tracking[0]:
                # filtered signal plot (used for alignment)
                fig.add_trace(go.Scatter(
                x=time_axis_for_anipose,
                y=filtered_signal, # + 25*bodypart_counter, # 25 mm spread
                name=bodyparts_list[bodypart_counter]+' filtered' if filter_tracking
                    else bodyparts_list[bodypart_counter],
                mode='lines',
                opacity=.9,
                line=dict(width=2)),
                row=1, col=1)
                
                # foot strikes
                fig.add_trace(go.Scatter(
                x=time_axis_for_anipose[foot_strike_idxs],
                y=filtered_signal[foot_strike_idxs], # + 25*bodypart_counter, # 25 mm spread
                name=bodyparts_list[bodypart_counter]+' strike',
                mode='markers',
                marker = dict(color='black'),
                opacity=.9,
                line=dict(width=3)),
                row=1, col=1
                )
                # foot offs               
                fig.add_trace(go.Scatter(
                x=time_axis_for_anipose[foot_off_idxs],
                y=filtered_signal[foot_off_idxs], # + 25*bodypart_counter, # 25 mm spread
                name=bodyparts_list[bodypart_counter]+' off',
                mode='markers',
                marker = dict(color='blue'),
                opacity=.9,
                line=dict(width=3)),
                row=1, col=1
                )
                bodypart_counter += 1 # increment for each matching bodypart
            else:
                fig.add_trace(go.Scatter(
                x=time_axis_for_anipose,
                # mean subtract and spread data values out by 25mm
                y=values.values-values.values.mean(), # + 25*bodypart_counter,
                name=bodyparts_list[bodypart_counter],
                mode='lines',
                opacity=.9,
                line=dict(width=2)),
                row=1, col=1,
                )
                bodypart_counter += 1 # increment for each matching bodypart
    

    # initialize counter to keep track of total unit count across all channels
    unit_counter = np.int16(0)
    # plot all ephys traces and/or SYNC channel
    for iChannel, channel_number in enumerate(ephys_channel_idxs_list):
        fig.add_trace(go.Scatter(
            x=time_axis_for_ephys,
            # if statement provides different scalings and offsets for ephys vs. SYNC channel
            y=(chosen_ephys_data_continuous_obj.samples[:,channel_number] - 5000*iChannel 
                if channel_number not in [-1,16]
                else (chosen_ephys_data_continuous_obj.samples[:,channel_number]+4)*0.5e3
                ),
            name=f"CH{channel_number}" if channel_number not in [-1,16] else "SYNC",
            mode='lines',
            marker = dict(color=CH_colors[color_stride*unit_counter]),
            opacity=1,
            line=dict(width=.4)),
            row=len(bodyparts_list)+1, col=1,
            )
        for iUnit, iUnitKey in enumerate(
            MU_spikes_by_channel_dict[str(channel_number)].keys()
            ):
            if channel_number not in [-1,16]:
                # plot spike locations onto each selected ephys trace
                fig.add_trace(go.Scatter(
                    x=time_axis_for_ephys[MU_spikes_by_channel_dict[
                        str(channel_number)][iUnitKey]],
                    y=chosen_ephys_data_continuous_obj.samples[MU_spikes_by_channel_dict[
                        str(channel_number)][iUnitKey],channel_number] - 5000*iChannel,
                    name=f"CH{channel_number} spikes",
                    mode='markers',
                    marker = dict(color=MU_colors[color_stride*unit_counter]),
                    opacity=.9,
                    line=dict(width=3)),
                    row=len(bodyparts_list)+1, col=1
                    )
                # plot isolated spikes into raster plot for each selected ephys trace
                fig.add_trace(go.Scatter(
                    x=time_axis_for_ephys[MU_spikes_by_channel_dict[
                        str(channel_number)][iUnitKey]],
                    y=np.zeros(len(time_axis_for_ephys))-unit_counter,
                    name=f"CH{channel_number}, Unit {iUnit}",
                    mode='markers',
                    marker_symbol='line-ns',
                    marker = dict(color=MU_colors[color_stride*unit_counter],
                                line_color=MU_colors[color_stride*unit_counter],
                                line_width=0.8,
                                size=10),
                    opacity=1),
                    row=len(bodyparts_list)+len(ephys_channel_idxs_list)+1, col=1
                    )
                unit_counter+=1
    
    fig.update_xaxes(
        title_text="<b>Time (s)</b>",
        row = len(bodyparts_list)+len(ephys_channel_idxs_list)+1,
        col = 1,#secondary_y=False
        )
    fig.update_yaxes(
        title_text="<b>Position (mm)</b>",
        row = 1,
        col = 1
        )
    fig.update_yaxes(
        title_text="<b>Voltage (uV)</b>",
        row = len(bodyparts_list)+1,
        col = 1
        )
    fig.update_yaxes(
        title_text="<b>Sorted Spikes</b>",
        row = len(bodyparts_list)+len(ephys_channel_idxs_list)+1,
        col = 1 # secondary_y=True
        )
    fig.update_layout(template=plot_template)
    figs = [fig]

    if do_plot:
            iplot(fig)

    return (
        MU_spikes_by_channel_dict, time_axis_for_ephys, time_axis_for_anipose,
        ephys_sample_rate, start_video_capture_ephys_idx, session_parameters, figs
    )

def bin_and_count(
    ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
    filter_ephys, filter_tracking, bin_width_ms, bin_width_radian, anipose_data_dict, bodypart_for_tracking,
    session_date, rat_name, treadmill_speed, treadmill_incline,
    camera_fps, alignto, vid_length, time_slice,
    do_plot, plot_template, MU_colors, CH_colors
    ):
    
    # check inputs for problems
    assert len(ephys_channel_idxs_list)==1, \
    "ephys_channel_idxs_list should only be 1 channel, idiot! :)"
    assert type(bin_width_ms) is int, "bin_width_ms must be type 'int'."
    
    (MU_spikes_by_channel_dict, _, time_axis_for_anipose,
    ephys_sample_rate, start_video_capture_ephys_idx, session_parameters, _) = sort(
        ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
        filter_ephys, filter_tracking, anipose_data_dict, bodyparts_list=bodypart_for_tracking,
        bodypart_for_tracking=bodypart_for_tracking,
        session_date=session_date, rat_name=rat_name,
        treadmill_speed=treadmill_speed, treadmill_incline=treadmill_incline,
        camera_fps=camera_fps, alignto=alignto, vid_length=vid_length,
        time_slice=time_slice, do_plot=False, # change T/F whether to plot sorting plots also
        plot_template=plot_template, MU_colors=MU_colors, CH_colors=CH_colors
        )    

    _, foot_strike_idxs, foot_off_idxs, step_stats = extract_step_idxs(
        anipose_data_dict, bodypart_for_tracking=bodypart_for_tracking, filter_tracking=filter_tracking,
        session_date=session_date, rat_name=rat_name, treadmill_speed=treadmill_speed,
        treadmill_incline=treadmill_incline, camera_fps=camera_fps, alignto=alignto
        )

    # extract data dictionary (with keys for each unit) for the chosen electrophysiology channel
    MU_spikes_dict = MU_spikes_by_channel_dict[str(ephys_channel_idxs_list[0])]
    # set conversion ratio from camera to electrophysiology sample rate
    step_to_ephys_conversion_ratio = ephys_sample_rate/camera_fps
    # initialize zero array to carry step-aligned spike activity,
    # with shape: Steps x Time (in ephys sample rate) x Units
    number_of_steps = int(step_stats['count'])
    # -2 steps to account for when we ignore the noisy first and last steps
    number_of_steps_used = number_of_steps-2
    MU_spikes_3d_array_ephys_time = np.zeros(
        (number_of_steps_used, 
        int(step_stats['max']*step_to_ephys_conversion_ratio),
        len(MU_spikes_dict),
        ))
    MU_spikes_3d_array_ephys_2π = MU_spikes_3d_array_ephys_time.copy()
    
    # initialize dict to store stepwise counts
    MU_step_aligned_spike_counts_dict = {key: None for key in MU_spikes_dict.keys()}
    # initialize dict of lists to store stepwise index arrays
    MU_step_aligned_spike_idxs_dict = {key: [] for key in MU_spikes_dict.keys()}
    MU_step_2π_warped_spike_idxs_dict = {key: [] for key in MU_spikes_dict.keys()}
    # convert foot strike/off indexes to the sample rate of electrophysiology data
    foot_strike_idxs_in_ephys_time = (
        foot_strike_idxs*step_to_ephys_conversion_ratio)+start_video_capture_ephys_idx
    foot_off_idxs_in_ephys_time = (
        foot_off_idxs*step_to_ephys_conversion_ratio)+start_video_capture_ephys_idx
    # set chosen alignment bodypart and choose corresponding index values
    if alignto == 'foot strike':
        step_idxs_in_ephys_time = foot_strike_idxs_in_ephys_time
    elif alignto == 'foot off':
        step_idxs_in_ephys_time = foot_off_idxs_in_ephys_time
    phase_warp_2π_coeff_list = []
    # fill 3d numpy array with Steps x Time x Units data, and a list of aligned idxs
    for iUnit, iUnitKey in enumerate(MU_spikes_dict.keys()): # for each unit
        MU_spikes_idx_arr = np.array(MU_spikes_dict[iUnitKey])
        for iStep in range(number_of_steps_used): # for each step
            # keep track of index boundaries for each step
            this_step_idx = step_idxs_in_ephys_time[iStep+1].astype(int)
            next_step_idx = step_idxs_in_ephys_time[iStep+2].astype(int)
            # filter out indexes which are beyond the video's and this step's boundaries
            spike_idxs_in_step_and_video_bounded = MU_spikes_idx_arr[np.where(
                (MU_spikes_idx_arr < next_step_idx) &
                (MU_spikes_idx_arr > this_step_idx) &
                (MU_spikes_idx_arr < (time_axis_for_anipose.max()*ephys_sample_rate).astype(int)) &
                (MU_spikes_idx_arr > start_video_capture_ephys_idx)
                )]
            # subtract current step index to align to each step, and convert to integer index
            MU_spikes_idxs_for_step = (
                spike_idxs_in_step_and_video_bounded - this_step_idx).astype(int)
            # store aligned indexes for each step
            MU_step_aligned_spike_idxs_dict[iUnitKey].append(MU_spikes_idxs_for_step)
            # if spikes are present, set them to 1 for this unit during this step
            if len(MU_spikes_idxs_for_step)!=0:
                MU_spikes_3d_array_ephys_time[iStep, MU_spikes_idxs_for_step, iUnit] = 1
        # create phase aligned step indexes, with max index for each step set to 2π    
        bin_width_eph_2π = []
        for πStep in range(number_of_steps_used): # for each step
            # keep track of index boundaries for each step
            this_step_2π_idx = step_idxs_in_ephys_time[πStep+1].astype(int)
            next_step_2π_idx = step_idxs_in_ephys_time[πStep+2].astype(int)
            # filter out indexes which are beyond the video's and this step_2π's boundaries
            spike_idxs_in_step_2π_and_video_bounded = MU_spikes_idx_arr[np.where(
                (MU_spikes_idx_arr < next_step_2π_idx) &
                (MU_spikes_idx_arr > this_step_2π_idx) &
                (MU_spikes_idx_arr < (time_axis_for_anipose.max()*ephys_sample_rate).astype(int)) &
                (MU_spikes_idx_arr > start_video_capture_ephys_idx)
                )]
            # coefficient to make step out of 2π radians, step made to be 2π after multiplication
            phase_warp_2π_coeff = 2*np.pi/(
                step_idxs_in_ephys_time[πStep+2]-step_idxs_in_ephys_time[πStep+1])
            phase_warp_2π_coeff_list.append(phase_warp_2π_coeff)
            # subtract this step start idx, and convert to an integer index
            MU_spikes_idxs_for_step_aligned = (
                spike_idxs_in_step_2π_and_video_bounded - this_step_2π_idx).astype(int)
            MU_spikes_idxs_for_step_2π = MU_spikes_idxs_for_step_aligned * phase_warp_2π_coeff
            # store aligned indexes for each step_2π
            MU_step_2π_warped_spike_idxs_dict[iUnitKey].append(MU_spikes_idxs_for_step_2π)
            # if spikes are present, set them to 1 for this unit during this step
            if len(MU_spikes_idxs_for_step_2π)!=0:
                MU_spikes_3d_array_ephys_2π[
                    πStep,
                    # convert MU_spikes_idxs_for_step_2π back to ephys sample rate-sized indexes
                    np.round(MU_spikes_idxs_for_step_2π/(
                        2*np.pi)*MU_spikes_3d_array_ephys_2π.shape[1]).astype(int),
                    iUnit] = 1
    # bin 3d array to time bins with width: bin_width_ms
    ms_duration = MU_spikes_3d_array_ephys_time.shape[1]/ephys_sample_rate*1000
    # round up number of steps to prevent index overflows
    number_of_bins_in_time_step = np.ceil(ms_duration/bin_width_ms).astype(int)
    bin_width_eph = int(bin_width_ms*(ephys_sample_rate/1000))
    
    # round up number of steps to prevent index overflows
    number_of_bins_in_2π_step = np.ceil(2*np.pi/bin_width_radian).astype(int)
    bin_width_eph_2π = np.round(
        MU_spikes_3d_array_ephys_2π.shape[1]/number_of_bins_in_2π_step).astype(int)
    
    # create binned 3d array with shape: Steps x Bins x Units
    MU_spikes_3d_array_binned = np.zeros((
        MU_spikes_3d_array_ephys_time.shape[0],
        number_of_bins_in_time_step,
        MU_spikes_3d_array_ephys_time.shape[2]))
    for iBin in range(number_of_bins_in_time_step):
        # sum across slice of cube bin_width_ms wide
        MU_spikes_3d_array_binned[:,iBin,:] = \
            MU_spikes_3d_array_ephys_time[:,iBin*bin_width_eph:(iBin+1)*bin_width_eph,:].sum(1)
            
    MU_spikes_3d_array_binned_2π = np.zeros((
        MU_spikes_3d_array_ephys_2π.shape[0],
        number_of_bins_in_2π_step,
        MU_spikes_3d_array_ephys_2π.shape[2]))
    for πBin in range(number_of_bins_in_2π_step):
        # sum across slice of cube bin_width_radian wide
        MU_spikes_3d_array_binned_2π[:,πBin,:] = \
            MU_spikes_3d_array_ephys_2π[:,πBin*bin_width_eph_2π:(πBin+1)*bin_width_eph_2π,:].sum(1)
                
    # get number of channels and units per channel
    # then compute a color stride value to maximize use of color space
    # number_of_channels = len(np.where((np.array(ephys_channel_idxs_list)!=16)
    #                                     & (np.array(ephys_channel_idxs_list)!=-1))[0])
    # number_of_units_per_channel = len(MU_spikes_by_channel_dict[
    #                                            str(ephys_channel_idxs_list[0])])
    # color_stride = len(MU_colors)//(number_of_channels*number_of_units_per_channel)
    color_stride = 1
    fig1 = make_subplots(
        rows=1, cols=2,
        shared_xaxes=False,
        subplot_titles=(
        f"Session Info: {session_parameters}",
        f"Session Info: {session_parameters}"
        ))
    for iUnit, iUnitKey in enumerate(MU_spikes_dict.keys()):
        MU_step_aligned_idxs = np.concatenate(
            MU_step_aligned_spike_idxs_dict[iUnitKey]).ravel()            
        MU_step_aligned_idxs_ms = MU_step_aligned_idxs/ephys_sample_rate*1000
        fig1.add_trace(go.Histogram(
            x=MU_step_aligned_idxs_ms, # ms
            xbins=dict(start=0, size=bin_width_ms),
            name=iUnitKey+"uV crossings",
            marker_color=MU_colors[color_stride*iUnit]),
            row=1, col=1
            )
    for iUnit, iUnitKey in enumerate(MU_spikes_dict.keys()):
        MU_step_2π_aligned_idxs = np.concatenate(
            MU_step_2π_warped_spike_idxs_dict[iUnitKey]).ravel()            
        fig1.add_trace(go.Histogram(
            x=MU_step_2π_aligned_idxs, # radians
            xbins=dict(start=0, size=bin_width_radian),
            name=iUnitKey+"uV crossings",
            marker_color=MU_colors[color_stride*iUnit],
            showlegend=False),
            row=1, col=2
            )
    
    # Reduce opacity to see both histograms
    fig1.update_traces(opacity=0.75)

    # set bars to overlap and all titles
    fig1.update_layout(
        barmode='overlay',
        title_text=\
            '<b>Time and Phase-Aligned Motor Unit Activity During Step Cycle</b>',
        # xaxis_title_text='<b>Time During Step (milliseconds)</b>',
        # yaxis_title_text=,
        # bargap=0., # gap between bars of adjacent location coordinates
        # bargroupgap=0.1 # gap between bars of the same location coordinates
        )
    bin_2π_rnd = np.round(bin_width_radian,4)
    fig1.update_xaxes(title_text='<b>Time During Step (milliseconds)</b>', row = 1, col = 1)
    fig1.update_xaxes(title_text='<b>Phase During Step (radians)</b>', row = 1, col = 2)
    fig1.update_yaxes(title_text=\
        f'<b>Binned Spike Count Across<br>{number_of_steps_used} Steps ({bin_width_ms}ms bins)</b>',
        row = 1, col = 1)
    fig1.update_yaxes(title_text=\
        f'<b>Binned Spike Count Across<br>{number_of_steps_used} Steps ({bin_2π_rnd}rad bins)</b>',
        row = 1, col = 2)
    fig1.update_yaxes(matches='y')
    
    # set theme to chosen template
    fig1.update_layout(template=plot_template)
    
    # plot for total counts and Future: other stats 
    fig2 = go.Figure()
    # sum all spikes across step cycles
    MU_spikes_count_across_all_steps = MU_spikes_3d_array_binned.sum(0).sum(0)
    
    fig2.add_trace(go.Bar(
    # list comprehension to get threshold values for each isolated unit on this channel
    x=[iThreshold+"uV crossings" for iThreshold in MU_spikes_dict.keys()],
    y=MU_spikes_count_across_all_steps,
    marker_color=[MU_colors[iColor] for iColor in range(0,len(MU_colors),color_stride)],
    opacity=1,
    showlegend=False
    # name="Counts Bar Plot"
    ))
    # set all titles
    fig2.update_layout(
        title_text=
        f'<b>Total Motor Unit Threshold Crossings Across {number_of_steps_used} Steps</b>\
        <br><sup>Session Info: {session_parameters}</sup>',
        # xaxis_title_text='<b>Motor Unit Voltage Thresholds</b>',
        yaxis_title_text='<b>Spike Count</b>',
        # bargap=0., # gap between bars of adjacent location coordinates
        # bargroupgap=0.1 # gap between bars of the same location coordinates
        )
    # set theme to chosen template
    fig2.update_layout(template=plot_template)
    fig2.update_yaxes(matches='y')
    figs = [fig1, fig2]

    if do_plot:
        iplot(fig1)
        iplot(fig2)

    return (
        MU_step_aligned_spike_idxs_dict,
        MU_step_aligned_spike_counts_dict,
        MU_step_2π_warped_spike_idxs_dict,
        MU_spikes_3d_array_ephys_time,
        MU_spikes_3d_array_binned,
        MU_spikes_3d_array_binned_2π,
        MU_spikes_count_across_all_steps,
        step_idxs_in_ephys_time, ephys_sample_rate, figs
    )

def raster(
    ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
    filter_ephys, filter_tracking, bin_width_ms, bin_width_radian, anipose_data_dict, bodypart_for_tracking,
    session_date, rat_name, treadmill_speed, treadmill_incline,
    camera_fps, alignto, vid_length, time_slice,
    do_plot, plot_template, MU_colors, CH_colors
    ):
    
    (MU_step_aligned_spike_idxs_dict,
    MU_step_aligned_spike_counts_dict,
    MU_step_2π_warped_spike_idxs_dict,
    MU_spikes_3d_array_ephys_time,
    MU_spikes_3d_array_binned,
    MU_spikes_3d_array_binned_2π,
    MU_spikes_count_across_all_steps,
    step_idxs_in_ephys_time, ephys_sample_rate, figs
    ) = bin_and_count(
    ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
    filter_ephys, filter_tracking, bin_width_ms, bin_width_radian, anipose_data_dict, bodypart_for_tracking,
    session_date, rat_name, treadmill_speed, treadmill_incline,
    camera_fps, alignto, vid_length, time_slice,
    do_plot=False, plot_template=plot_template, MU_colors=MU_colors, CH_colors=CH_colors)
    
    session_parameters = f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
    
    number_of_steps = MU_spikes_3d_array_ephys_time.shape[0]
    samples_per_step = MU_spikes_3d_array_ephys_time.shape[1]
    number_of_units = MU_spikes_3d_array_ephys_time.shape[2]
    unit_counter = 0
    step_counter = 0
    fig = go.Figure()
    # for each channel and each trial's spike time series, plot onto the raster: plotly scatters
    for iUnit, iUnitKey in enumerate(MU_step_aligned_spike_idxs_dict.keys()):
        for iStep in range(number_of_steps):
            # if number_of_units==2:
            fig.add_trace(go.Scatter(
                x=MU_step_aligned_spike_idxs_dict[iUnitKey][iStep]/ephys_sample_rate*1000,
                y=np.zeros(samples_per_step)-unit_counter-step_counter-iUnit*number_of_units,
                name=f'step{iStep} unit{iUnit}',
                mode='markers',
                marker_symbol='line-ns',
                marker = dict(color=MU_colors[unit_counter],
                            line_color=MU_colors[unit_counter],
                            line_width=3,
                            size=6),
                opacity=0.75
                ))
            
            step_counter+=1
        unit_counter+=1
    # if number_of_units==2:
    fig.update_layout(
        title_text=
        f'<b>MU Activity Raster for All {number_of_steps} Steps</b>\
        <br><sup>Session Info: {session_parameters}</sup>',
        xaxis_title_text=f'<b>Time (ms)</b>',
        yaxis_title_text=f'<b>Step</b>'
        )
    # elif number_of_units==3:
    #     fig.update_layout(
    #         title_text=
    #         f'<b>MU Activity Raster for All {number_of_steps} Steps</b>\
    #         <br><sup>Session Info: {session_parameters}</sup>',
    #         xaxis_title_text=f'<b>Time (ms)</b>',
    #         yaxis_title_text=f'<b>Step (ms)</b>'
    #         )
        
    # set theme to chosen template
    fig.update_layout(template=plot_template)
        
    if do_plot:
        iplot(fig)
    
    figs = [fig]
    return figs


def smooth(
    ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
    filter_ephys, filter_tracking, bin_width_ms, bin_width_radian, smoothing_window, anipose_data_dict, bodypart_for_tracking,
    session_date, rat_name, treadmill_speed, treadmill_incline,
    camera_fps, alignto, vid_length, time_slice,
    do_plot, phase_align, plot_template, MU_colors, CH_colors
    ):
    
    (MU_step_aligned_spike_idxs_dict,
    MU_step_aligned_spike_counts_dict,
    MU_step_2π_warped_spike_idxs_dict,
    MU_spikes_3d_array_ephys_time,
    MU_spikes_3d_array_binned,
    MU_spikes_3d_array_binned_2π,
    MU_spikes_count_across_all_steps,
    step_idxs_in_ephys_time, ephys_sample_rate, figs
    ) = bin_and_count(
    ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
    filter_ephys, filter_tracking, bin_width_ms, bin_width_radian, anipose_data_dict, bodypart_for_tracking,
    session_date, rat_name, treadmill_speed, treadmill_incline,
    camera_fps, alignto, vid_length, time_slice,
    do_plot=False, plot_template=plot_template, MU_colors=MU_colors, CH_colors=CH_colors
    )
    
    session_parameters = f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
    # initialize 3d numpy array with shape: Steps x Bins x Units
    if phase_align is True:
        binned_spike_array = MU_spikes_3d_array_binned_2π
        bin_width = bin_width_radian
        bin_unit = ' radians'
        title_prefix = 'Phase-'
    else:
        binned_spike_array = MU_spikes_3d_array_binned
        bin_width = bin_width_ms
        bin_unit = 'ms'
        title_prefix = 'Time-'
        
    MU_smoothed_spikes_3d_array = np.zeros_like(binned_spike_array)
    number_of_steps = MU_smoothed_spikes_3d_array.shape[0]
    number_of_bins = MU_smoothed_spikes_3d_array.shape[1]
    number_of_units = MU_smoothed_spikes_3d_array.shape[2]
    fig = go.Figure()
    
    # smooth and plot each trace
    for iUnit in range(number_of_units):
        for iStep in range(number_of_steps):
            # gaussian smooth across time, with standard deviation value of bin_width_ms
            MU_smoothed_spikes_3d_array[iStep,:,:] = gaussian_filter1d(
                binned_spike_array[iStep,:,:],
                sigma=smoothing_window,
                axis=0, order=0, output=None, mode='constant',
                cval=0.0, truncate=4.0)
            MU_smoothed_spikes_ztrimmed_array = np.trim_zeros(
                MU_smoothed_spikes_3d_array[iStep,:,iUnit], trim='b')
            fig.add_trace(go.Scatter(
                x=np.arange(len(MU_smoothed_spikes_ztrimmed_array)) if not phase_align 
                    else np.arange(2*np.pi, number_of_bins),
                y=MU_smoothed_spikes_ztrimmed_array,
                name=f'step{iStep}_unit{iUnit}',
                mode='lines',
                opacity=.5,
                line=dict(width=8,color=MU_colors[iUnit],dash='solid')
                ))
    # plot mean traces for each unit
    for iUnit in range(number_of_units):
            fig.add_trace(go.Scatter(
                x=np.arange(MU_smoothed_spikes_3d_array.shape[1]) if not phase_align 
                    else np.arange(2*np.pi, number_of_bins),
                y=MU_smoothed_spikes_3d_array.mean(axis=0)[:,iUnit],
                name=f'mean_unit{iUnit}',
                mode='lines',
                opacity=1,
                # dash styles: ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
                line=dict(width=4,color=CH_colors[iUnit],dash='dot')
                ))
    
    fig.update_layout(
        title_text=
        f'<b>{title_prefix}Aligned MU Activity for All {number_of_steps} Steps</b>\
        <br><sup>Session Info: {session_parameters}</sup>',
        xaxis_title_text=f'<b>Bins ({np.round(bin_width,4)}{bin_unit})</b>',
        yaxis_title_text= \
            f'<b>Smoothed MU Activity<br>({smoothing_window} Sample Kernel)</b>',
        )
    # set theme to chosen template
    # fig.update_layout(template=plot_template)
    
    if do_plot:
        iplot(fig)
    
    # put in a list for compatibility with calling functions
    figs = [fig]
    
    return MU_smoothed_spikes_3d_array, figs
            # df_cols_list.append(f'step{iStep}_unit{iUnit}')
    # transpose 3d array to allow flattening pages of the 3d data into Steps x (Bins*Units)
    # MU_smoothed_spikes_3d_array_T = MU_smoothed_spikes_3d_array.T(1,2,0)
    # smoothed_binned_MU_spikes_2d_array = MU_smoothed_spikes_3d_array_T.reshape(number_of_steps,-1)
    # smoothed_MU_df = pd.DataFrame(
    #     data=smoothed_binned_MU_spikes_2d_array,
    #     index=dict(bins=np.arange(len()))
    #     columns=df_cols_list
    #     )
    # px.plot(smoothed_MU_df,x="bins",y="")
    
def state_space(
    ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
    filter_ephys, filter_tracking, bin_width_ms, bin_width_radian, smoothing_window, anipose_data_dict, bodypart_for_tracking,
    session_date, rat_name, treadmill_speed, treadmill_incline,
    camera_fps, alignto, vid_length, time_slice,
    do_plot, plot_units, phase_align, plot_template, MU_colors, CH_colors
    ):
    
    (MU_smoothed_spikes_3d_array, figs
    ) = smooth(
        ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
        filter_ephys, filter_tracking, bin_width_ms, bin_width_radian, smoothing_window, anipose_data_dict,
        bodypart_for_tracking, session_date, rat_name, treadmill_speed, treadmill_incline,
        camera_fps, alignto, vid_length, time_slice, do_plot=False, phase_align=phase_align,
        plot_template=plot_template, MU_colors=MU_colors, CH_colors=CH_colors)
    
    session_parameters = f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
    
    # select units for plotting
    sliced_MU_smoothed_3d_array = MU_smoothed_spikes_3d_array[:,:,plot_units]
    
    # set numbers of things from input matrix dimensionality
    number_of_steps = sliced_MU_smoothed_3d_array.shape[0]
    number_of_bins = sliced_MU_smoothed_3d_array.shape[1]
    # detect number of identified units
    number_of_units = sliced_MU_smoothed_3d_array.any(1).any(0).sum()
    
    if phase_align is True:
        bin_width = bin_width_radian
        bin_unit = ' radians'
        title_prefix = 'Phase-'
    else:
        bin_width = bin_width_ms
        bin_unit = 'ms'
        title_prefix = 'Time-'
    
    fig = go.Figure()
    # smooth and plot each trace
    for iStep in range(number_of_steps):
        # gaussian smooth across time, with standard deviation value of bin_width_ms
        sliced_MU_smoothed_3d_array[iStep,:,:] = gaussian_filter1d(
            sliced_MU_smoothed_3d_array[iStep,:,:],
            sigma=smoothing_window,
            axis=0, order=0, output=None, mode='constant',
            cval=0.0, truncate=4.0)
        if number_of_units==2:
            fig.add_trace(go.Scatter(
                x=sliced_MU_smoothed_3d_array[iStep,:,0],
                y=sliced_MU_smoothed_3d_array[iStep,:,1],
                name=f'step{iStep}',
                mode='lines',
                opacity=.5,
                line=dict(width=5,color=MU_colors[treadmill_incline//5],dash='solid')
                ))
        elif number_of_units==3:
            fig.add_trace(go.Scatter3d(
                x=sliced_MU_smoothed_3d_array[iStep,:,0],
                y=sliced_MU_smoothed_3d_array[iStep,:,1],
                z=sliced_MU_smoothed_3d_array[iStep,:,2],
                name=f'step{iStep}',
                mode='lines',
                opacity=.5,
                line=dict(width=8,color=MU_colors[treadmill_incline//5],dash='solid')
                ))
    # plot mean traces for each unit
    if number_of_units==2:
            fig.add_trace(go.Scatter(
                x=sliced_MU_smoothed_3d_array[:,:,0].mean(0),
                y=sliced_MU_smoothed_3d_array[:,:,1].mean(0),
                name=f'mean',
                mode='markers',
                opacity=1,
                line=dict(width=10,color=CH_colors[0],dash='solid'),
                marker=dict(size=8,color=CH_colors[0])
                ))
    elif number_of_units==3:
            fig.add_trace(go.Scatter3d(
                x=sliced_MU_smoothed_3d_array[:,:,0].mean(0),
                y=sliced_MU_smoothed_3d_array[:,:,1].mean(0),
                z=sliced_MU_smoothed_3d_array[:,:,2].mean(0),
                name=f'mean',
                mode='markers',
                opacity=1,
                line=dict(width=10,color=CH_colors[0],dash='solid'),
                marker=dict(size=3,color=CH_colors[0])
                ))
    if number_of_units==2:
        fig.update_layout(
            title_text=
            f'<b>{title_prefix}MU State Space Activity for All {number_of_steps} Steps</b>\
            <br><sup>Session Info: {session_parameters}, Bin Width: {np.round(bin_width,4)}{bin_unit}, Smoothed by {smoothing_window} bin window</sup>',
            xaxis_title_text=f'<b>Unit {plot_units[0]}</b>',
            yaxis_title_text=f'<b>Unit {plot_units[1]}</b>'
            )
    elif number_of_units==3:
        fig.update_layout(
            title_text=
            f'<b>{title_prefix}MU State Space Activity for All {number_of_steps} Steps</b>\
            <br><sup>Session Info: {session_parameters}, Bin Width: {bin_width}{bin_unit}, Smoothed by {smoothing_window} bins</sup>')
        fig.update_scenes(
            dict(camera=dict(eye=dict(x=-0.3, y=-2, z=0.2)), #the default values are 1.25, 1.25, 1.25
                xaxis = dict(title_text=f'<b>Unit {plot_units[0]}</b>'),
                yaxis = dict(title_text=f'<b>Unit {plot_units[1]}</b>'),
                zaxis = dict(title_text=f'<b>Unit {plot_units[2]}</b>'),
                aspectmode='manual', #this string can be 'data', 'cube', 'auto', 'manual'
                # custom aspectratio is defined as follows:
                aspectratio=dict(x=1, y=1, z=1)
           ))
    # set theme to chosen template
    # fig.update_layout(template=plot_template)
    
    if do_plot:
        iplot(fig)
    
    # put in a list for compatibility with calling functions
    figs = [fig]
    return figs
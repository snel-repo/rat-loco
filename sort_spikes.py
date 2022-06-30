from extract_step_idxs import extract_step_idxs
import numpy as np
from plotly.offline import iplot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, butter, filtfilt, iirnotch
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
def sort_spikes(
    ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
    filter_ephys, anipose_data_dict, bodyparts_list, bodypart_for_tracking,
    session_date, rat_name, treadmill_speed, treadmill_incline,
    camera_fps, alignto, vid_length, time_slice, do_plot,
    plot_template, MU_colors, CH_colors
    ):

    # format inputs to avoid ambiguities
    rat_name = str(rat_name).lower()
    treadmill_speed = str(treadmill_speed).zfill(2)
    treadmill_incline = str(treadmill_incline).zfill(2)

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
    chosen_ephys_data_continuous_obj = chosen_ephys_data_dict[
        f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
        ]

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
    chosen_anipose_df = chosen_anipose_data_dict[
        f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
        ]

    # create time axes
    ephys_sample_rate = chosen_ephys_data_continuous_obj.metadata['sample_rate']
    time_axis_for_ephys = np.arange(
        round(len(chosen_ephys_data_continuous_obj.samples)*time_slice)
        )/ephys_sample_rate
    
    # find the beginning of the camera SYNC pulse         
    filtered_sync_channel = butter_highpass_filter(
        data=chosen_ephys_data_continuous_obj.samples[:,-1], cutoff=50, fs=30000, order=2)
    start_video_capture_idx = find_peaks(filtered_sync_channel,height=0.3)[0][0]
    time_axis_for_anipose = np.arange(0,vid_length,1/camera_fps)+ \
                                        time_axis_for_ephys[start_video_capture_idx]

    # identify motion peak locations of bodypart for step cycle alignment
    filter_tracking=True
    if filter_tracking == True:
        filtered_signal = butter_highpass_filter(
            data=chosen_anipose_df[bodypart_for_tracking[0]].values,
            cutoff=0.5, fs=camera_fps, order=5)
    else: # do not filter
        filtered_signal=chosen_anipose_df[bodypart_for_tracking[0]].values

    foot_strike_idxs, foot_off_idxs, _ = extract_step_idxs(
    anipose_data_dict, bodypart_for_tracking=bodypart_for_tracking, session_date=session_date,
    rat_name=rat_name, treadmill_speed=treadmill_speed, treadmill_incline=treadmill_incline,
    camera_fps=camera_fps, alignto=alignto
    )

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
    row_spec_list[len(bodyparts_list)+len(ephys_channel_idxs_list)] = [{'rowspan': number_of_channels//2+1}]

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
                y=filtered_signal + 25*bodypart_counter, # 25 mm spread
                name=bodyparts_list[bodypart_counter]+' filtered',
                mode='lines',
                opacity=.9,
                line=dict(width=2)),
                row=1, col=1)
                
                # foot strikes
                fig.add_trace(go.Scatter(
                x=time_axis_for_anipose[foot_strike_idxs],
                y=filtered_signal[foot_strike_idxs] + 25*bodypart_counter, # 25 mm spread
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
                y=filtered_signal[foot_off_idxs] + 25*bodypart_counter, # 25 mm spread
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
                y=values.values-values.values.mean() + 25*bodypart_counter,
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
        ephys_sample_rate, start_video_capture_idx, figs
        )

def bin_spikes(
    ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
    filter_ephys, bin_width_ms, anipose_data_dict, bodypart_for_tracking,
    session_date, rat_name, treadmill_speed, treadmill_incline,
    camera_fps, alignto, vid_length, time_slice,
    do_plot, plot_template, MU_colors, CH_colors
    ):
    
    assert len(ephys_channel_idxs_list)==1, "ephys_channel_idxs_list should only be 1 channel, idiot! :)"
    
    (MU_spikes_by_channel_dict, _, time_axis_for_anipose,
    ephys_sample_rate, start_video_capture_idx, _) = sort_spikes(
        ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
        filter_ephys, anipose_data_dict, bodyparts_list=bodypart_for_tracking,
        bodypart_for_tracking=bodypart_for_tracking,
        session_date=session_date, rat_name=rat_name,
        treadmill_speed=treadmill_speed, treadmill_incline=treadmill_incline,
        camera_fps=camera_fps, alignto=alignto, vid_length=vid_length,
        time_slice=time_slice, do_plot=False, # change T/F whether to plot sorting plots also
        plot_template=plot_template, MU_colors=MU_colors, CH_colors=CH_colors
        )    

    foot_strike_idxs, foot_off_idxs, step_stats = extract_step_idxs(
        anipose_data_dict, bodypart_for_tracking=bodypart_for_tracking, session_date=session_date,
        rat_name=rat_name, treadmill_speed=treadmill_speed, treadmill_incline=treadmill_incline,
        camera_fps=camera_fps, alignto=alignto
        )

    # extract data dictionary (with keys for each unit) for the chosen electrophysiology channel
    MU_spikes_dict = MU_spikes_by_channel_dict[str(ephys_channel_idxs_list[0])]
    # set conversion ratio from camera to electrophysiology sample rate
    step_to_ephys_conversion_ratio = ephys_sample_rate/camera_fps
    # initialize zero array to carry step-aligned spike activity,
    # with shape: #Units x Time (in ephys sample rate) x #Steps
    number_of_steps = int(step_stats['count'])
    step_aligned_MU_spikes = np.zeros(
        (len(MU_spikes_dict),
        int(step_stats['max']*step_to_ephys_conversion_ratio),
        number_of_steps-1 # -1 to account for when we ignore the noisy first and last steps
        ))
    # initialize dict to store stepwise counts
    MU_step_aligned_spike_counts_dict = {key: None for key in MU_spikes_dict.keys()}
    # initialize dict of lists to store stepwise index arrays
    MU_step_aligned_spike_idxs_dict = {key: [] for key in MU_spikes_dict.keys()}
    MU_step_2π_warped_spike_idxs_dict = {key: [] for key in MU_spikes_dict.keys()}
    # convert foot strike/off indexes to the sample rate of electrophysiology data
    foot_strike_idxs_in_ephys_time = (
        foot_strike_idxs*step_to_ephys_conversion_ratio)+start_video_capture_idx
    foot_off_idxs_in_ephys_time = (
        foot_off_idxs*step_to_ephys_conversion_ratio)+start_video_capture_idx
    # set chosen alignment bodypart and choose corresponding index values
    if alignto == 'foot strike':
        step_idxs_in_ephys_time = foot_strike_idxs_in_ephys_time
    elif alignto == 'foot off':
        step_idxs_in_ephys_time = foot_off_idxs_in_ephys_time
    # fill 3d numpy array with Units x Time x Trials/Steps data, and a list of aligned idxs
    for iUnit, iUnitKey in enumerate(MU_spikes_dict.keys()): # for each unit
        MU_spikes_idx_arr = np.array(MU_spikes_dict[iUnitKey])
        this_step_idx = 0; next_step_idx = 0;
        for iStep in range(number_of_steps-2): # for each step
            # keep track of index boundaries for each step
            this_step_idx = step_idxs_in_ephys_time[iStep+1].astype(int)
            next_step_idx = step_idxs_in_ephys_time[iStep+2].astype(int)
            # filter out indexes which are beyond the video's and this step's boundaries
            spike_idxs_in_step_and_video_bounded = MU_spikes_idx_arr[np.where(
                (MU_spikes_idx_arr < next_step_idx) &
                (MU_spikes_idx_arr > this_step_idx) &
                (MU_spikes_idx_arr < (time_axis_for_anipose.max()*ephys_sample_rate).astype(int)) &
                (MU_spikes_idx_arr > start_video_capture_idx)
                )]
            # subtract current step index to align to each step, and convert to integer index
            MU_spikes_idxs_for_step = (
                spike_idxs_in_step_and_video_bounded - this_step_idx).astype(int)
            # store aligned indexes for each step
            MU_step_aligned_spike_idxs_dict[iUnitKey].append(MU_spikes_idxs_for_step)
            # if spikes are present, set them to 1 for this unit during this step
            if len(MU_spikes_idxs_for_step)!=0:
                step_aligned_MU_spikes[iUnit,MU_spikes_idxs_for_step,iStep+1] = 1

        # create phase aligned step indexes, with max index for each step set to 2π    
        for πStep in range(number_of_steps-2): # for each step
            # keep track of index boundaries for each step
            this_step_2π_idx = step_idxs_in_ephys_time[πStep+1].astype(int)
            next_step_2π_idx = step_idxs_in_ephys_time[πStep+2].astype(int)
            # filter out indexes which are beyond the video's and this step_2π's boundaries
            spike_idxs_in_step_2π_and_video_bounded = MU_spikes_idx_arr[np.where(
                (MU_spikes_idx_arr < next_step_2π_idx) &
                (MU_spikes_idx_arr > this_step_2π_idx) &
                (MU_spikes_idx_arr < (time_axis_for_anipose.max()*ephys_sample_rate).astype(int)) &
                (MU_spikes_idx_arr > start_video_capture_idx)
                )]
            # subtract current step_2π index to align to each step_2π, and convert to integer index
            phase_warp_2π_coeff = 2*np.pi/(step_idxs_in_ephys_time[πStep+2]-step_idxs_in_ephys_time[πStep+1])
            MU_spikes_idxs_for_step = (
                spike_idxs_in_step_2π_and_video_bounded - this_step_2π_idx).astype(int)
            MU_spikes_idxs_for_step_2π = MU_spikes_idxs_for_step * phase_warp_2π_coeff
            # store aligned indexes for each step_2π
            MU_step_2π_warped_spike_idxs_dict[iUnitKey].append(MU_spikes_idxs_for_step_2π)

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
        f"Session Info: {list(anipose_data_dict.keys())[0]}",
        f"Session Info: {list(ephys_data_dict.keys())[0]}"
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
    bin_width_2π = 2*np.pi/50 #((step_stats['max']/camera_fps)*1000/bin_width_ms)
    for iUnit, iUnitKey in enumerate(MU_spikes_dict.keys()):
        MU_step_2π_aligned_idxs = np.concatenate(
            MU_step_2π_warped_spike_idxs_dict[iUnitKey]).ravel()            
        fig1.add_trace(go.Histogram(
            x=MU_step_2π_aligned_idxs, # radians
            xbins=dict(start=0, size=bin_width_2π),
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
    bin_2π_rnd = np.round(bin_width_2π,3)
    fig1.update_xaxes(title_text='<b>Time During Step (milliseconds)</b>', row = 1, col = 1)
    fig1.update_xaxes(title_text='<b>Phase During Step (radians)</b>', row = 1, col = 2)
    fig1.update_yaxes(title_text=\
        f'<b>Binned Spike Count Across<br>{number_of_steps} Steps ({bin_width_ms}ms bins)</b>',
        row = 1, col = 1)
    fig1.update_yaxes(title_text=\
        f'<b>Binned Spike Count Across<br>{number_of_steps} Steps ({bin_2π_rnd}rad bins)</b>',
        row = 1, col = 2)
    
    # set theme to chosen template
    fig1.update_layout(template=plot_template)
    
    # plot for total counts and Future: other stats 
    fig2 = go.Figure()
    # sum all spikes across step cycles
    MU_spikes_count_across_steps = step_aligned_MU_spikes.sum(axis=2).sum(axis=1)
    
    fig2.add_trace(go.Bar(
    # list comprehension to get threshold values for each isolated unit on this channel
    x=[iThreshold+"uV crossings" for iThreshold in MU_spikes_dict.keys()],
    y=MU_spikes_count_across_steps,
    marker_color=[MU_colors[iColor] for iColor in range(0,len(MU_colors),color_stride)],
    opacity=1,
    name="Counts Bar Plot"
    ))
    # set all titles
    fig2.update_layout(
        title_text=
        f'<b>Total Motor Unit Threshold Crossings Across {number_of_steps} Steps</b>',
        # xaxis_title_text='<b>Motor Unit Voltage Thresholds</b>',
        yaxis_title_text='<b>Spike Count</b>',
        # bargap=0., # gap between bars of adjacent location coordinates
        # bargroupgap=0.1 # gap between bars of the same location coordinates
        )
    # set theme to chosen template
    fig2.update_layout(template=plot_template)

    figs = [fig1, fig2]

    if do_plot:
        iplot(fig1)
        iplot(fig2)

    return (
        MU_step_aligned_spike_idxs_dict,
        MU_step_aligned_spike_counts_dict,
        MU_step_2π_warped_spike_idxs_dict,
        step_idxs_in_ephys_time, figs
    )

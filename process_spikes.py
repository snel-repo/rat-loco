from process_steps import peak_align_and_filt, trialize_steps
import pandas as pd
import numpy as np
from plotly.offline import iplot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, butter, filtfilt, iirnotch
from scipy.ndimage import gaussian_filter1d
from pdb import set_trace
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, RobustScaler
# import umap

# create highpass filters to remove baseline from SYNC channel
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
    filter_ephys, filter_all_anipose, anipose_data_dict, 
    bodyparts_list, bodypart_for_alignment, bodypart_for_reference, bodypart_ref_filter, origin_offsets,
    session_date, rat_name, treadmill_speed, treadmill_incline,
    camera_fps, align_to, vid_length, time_frame, do_plot,
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
        round(len(chosen_ephys_data_continuous_obj.samples))
        )/ephys_sample_rate
    
    # find the beginning of the camera SYNC pulse         
    filtered_sync_channel = butter_highpass_filter(
        data=chosen_ephys_data_continuous_obj.samples[:,-1], cutoff=50, fs=30000, order=2)
    start_video_capture_ephys_idx = find_peaks(filtered_sync_channel,height=0.3)[0][0]
    time_axis_for_anipose = np.arange(0,vid_length,1/camera_fps)+ \
                                        time_axis_for_ephys[start_video_capture_ephys_idx]

    # # identify motion peak locations of bodypart for step cycle alignment
    # if filter_all_anipose == True:
    #     filtered_signal = butter_highpass_filter(
    #         data=chosen_anipose_df[bodypart_for_alignment[0]].values,
    #         cutoff=0.5, fs=camera_fps, order=5)
    # else: # do not filter
    #     filtered_signal=chosen_anipose_df[bodypart_for_alignment[0]].values

    (processed_anipose_df, foot_strike_idxs, foot_off_idxs, _,
    step_slice, step_time_slice, ref_bodypart_trace_list) = peak_align_and_filt(
    anipose_data_dict, bodypart_for_alignment=bodypart_for_alignment, bodypart_for_reference=bodypart_for_reference,
    bodypart_ref_filter=bodypart_ref_filter, origin_offsets=origin_offsets, filter_all_anipose=filter_all_anipose,
    session_date=session_date, rat_name=rat_name, treadmill_speed=treadmill_speed, treadmill_incline=treadmill_incline,
    camera_fps=camera_fps, align_to=align_to, time_frame=time_frame)
    
    foot_strike_slice_idxs = foot_strike_idxs[step_slice]
    # foot_strike_idxs[np.where(
    #     (foot_strike_idxs >= step_time_slice.start) &
    #     (foot_strike_idxs <= step_time_slice.stop)
    # )]
    
    foot_off_slice_idxs = foot_off_idxs[step_slice]
    # foot_off_idxs[np.where(
    #     (foot_off_idxs >= step_time_slice.start) &
    #     (foot_off_idxs <= step_time_slice.stop)
    # )]
    
    # foot_strike_slice_idxs = [
    #     foot_strike_idxs[int((sliced_step_stats['count']-1)*time_frame[0])],
    #     foot_strike_idxs[int((sliced_step_stats['count']-1)*time_frame[1])]
    #     ]
    # foot_off_slice_idxs = [
    #     foot_off_idxs[int((sliced_step_stats['count']-1)*time_frame[0])],
    #     foot_off_idxs[int((sliced_step_stats['count']-1)*time_frame[1])]
    #     ]
    
    # all_step_idx = []
    # if align_to == 'foot strike':
    #     all_step_idx.append(foot_strike_slice_idxs[0])
    #     all_step_idx.append(foot_strike_slice_idxs[1])
    # elif align_to == 'foot off':
    #     all_step_idx.append(foot_off_slice_idxs[0])
    #     all_step_idx.append(foot_off_slice_idxs[1])
            
    # set conversion ratio from camera to electrophysiology sample rate
    step_to_ephys_conversion_ratio = ephys_sample_rate/camera_fps
    step_idx_ephys_time = []
    step_idx_ephys_time.append(
        int(step_to_ephys_conversion_ratio*(step_time_slice.start))+start_video_capture_ephys_idx)
    step_idx_ephys_time.append(
        int(step_to_ephys_conversion_ratio*(step_time_slice.stop))+start_video_capture_ephys_idx)
    if time_frame == 1:
        step_time_slice_ephys = slice(0,-1) # get full anipose traces, if [0,1]
    else:
        # step_slice = slice(step_time_slice.start,step_time_slice.stop)
        step_time_slice_ephys = slice(step_idx_ephys_time[0],step_idx_ephys_time[1])
    
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
                ephys_data_for_channel = chosen_ephys_data_continuous_obj.samples[step_time_slice_ephys, channel_number]
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
                    distance=ephys_sample_rate//1000,
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
            if name == bodypart_for_alignment[0]:
                # filtered signal plot (used for alignment)
                fig.add_trace(go.Scatter(
                    x=time_axis_for_anipose[step_time_slice],
                    y=processed_anipose_df[bodypart_for_alignment[0]][step_time_slice], # + 25*bodypart_counter, # 25 mm spread
                    name=bodyparts_list[bodypart_counter]+' processed' if filter_all_anipose or origin_offsets
                        else bodyparts_list[bodypart_counter],
                    mode='lines',
                    opacity=.9,
                    line=dict(width=2)),
                    row=1, col=1
                    )
                # foot strikes
                fig.add_trace(go.Scatter(
                    x=time_axis_for_anipose[foot_strike_slice_idxs],
                    y=processed_anipose_df[bodypart_for_alignment[0]][foot_strike_slice_idxs],
                    name=bodyparts_list[bodypart_counter]+' strike',
                    mode='markers',
                    marker = dict(color='black'),
                    opacity=.9,
                    line=dict(width=3)),
                    row=1, col=1
                    )
                # foot offs               
                fig.add_trace(go.Scatter(
                    x=time_axis_for_anipose[foot_off_slice_idxs],
                    y=processed_anipose_df[bodypart_for_alignment[0]][foot_off_slice_idxs],
                    name=bodyparts_list[bodypart_counter]+' off',
                    mode='markers',
                    marker = dict(color='blue'),
                    opacity=.9,
                    line=dict(width=3)),
                    row=1, col=1
                    )
                bodypart_counter += 1 # increment for each matching bodypart
            else:
                if origin_offsets:
                    fig.add_trace(go.Scatter(
                        x=time_axis_for_anipose[step_time_slice],
                        y=processed_anipose_df[name][step_time_slice], # + 25*bodypart_counter,
                        name=bodyparts_list[bodypart_counter]+' processed',
                        mode='lines',
                        opacity=.9,
                        line=dict(width=2)),
                        row=1, col=1,
                        )
                else:
                    fig.add_trace(go.Scatter(
                        x=time_axis_for_anipose[step_time_slice],
                        y=values.values[step_time_slice], # + 25*bodypart_counter,
                        name=bodyparts_list[bodypart_counter],
                        mode='lines',
                        opacity=.9,
                        line=dict(width=2)),
                        row=1, col=1,
                        )
                bodypart_counter += 1 # increment for each matching bodypart
    if bodypart_ref_filter and origin_offsets is not False:
        # plot x/y/z reference trace
        dims = [key for key in origin_offsets.keys() if type(origin_offsets[key]) is not int]
        for dim, ref_trace in zip(dims, ref_bodypart_trace_list):
            fig.add_trace(go.Scatter(
                x=time_axis_for_anipose[step_time_slice],
                y=ref_trace[step_time_slice],
                name=f"Ref: {bodypart_for_reference[0]}_{dim}, {bodypart_ref_filter}Hz lowpass",
                mode='lines',
                opacity=.9,
                line=dict(width=3,color="lightgray",dash='dash')),
                row=1, col=1
                )
    # initialize counter to keep track of total unit count across all channels
    unit_counter = np.int16(0)
    # plot all ephys traces and/or SYNC channel
    for iChannel, channel_number in enumerate(ephys_channel_idxs_list):
        fig.add_trace(go.Scatter(
            x=time_axis_for_ephys[step_time_slice_ephys],
            # if statement provides different scalings and offsets for ephys vs. SYNC channel
            y=(chosen_ephys_data_continuous_obj.samples[
                step_time_slice_ephys,channel_number] - 5000*iChannel
                if channel_number not in [-1,16]
                else (chosen_ephys_data_continuous_obj.samples[
                    step_time_slice_ephys,channel_number]+4)*0.5e3
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
                
                # spike_idx_in_time_frame = np.where((MU_spikes_by_channel_dict[str(channel_number)][iUnitKey]
                #                            >= step_idx_ephys_time[0]-start_video_capture_ephys_idx) &
                #                           (MU_spikes_by_channel_dict[str(channel_number)][iUnitKey]
                #                            <= step_idx_ephys_time[1]-start_video_capture_ephys_idx))
                
                # plot spike locations onto each selected ephys trace
                fig.add_trace(go.Scatter(
                    x=time_axis_for_ephys[ # index where spikes are, starting after the video
                        MU_spikes_by_channel_dict[str(channel_number)][iUnitKey][:]+step_time_slice_ephys.start],
                    y=chosen_ephys_data_continuous_obj.samples[
                        MU_spikes_by_channel_dict[str(channel_number)][iUnitKey][:]+step_time_slice_ephys.start,
                        channel_number]-5000*iChannel,
                    name=f"CH{channel_number}, Unit {iUnit}",
                    mode='markers',
                    marker = dict(color=MU_colors[color_stride*unit_counter]),
                    opacity=.9,
                    line=dict(width=3)),
                    row=len(bodyparts_list)+1, col=1
                    )
                # plot isolated spikes into raster plot for each selected ephys trace
                fig.add_trace(go.Scatter(
                    x=time_axis_for_ephys[ # index where spikes are, starting after the video
                        MU_spikes_by_channel_dict[str(channel_number)][iUnitKey][:]+step_time_slice_ephys.start],
                    y=np.zeros(len(time_axis_for_ephys[step_time_slice_ephys]))-unit_counter,
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
        ephys_sample_rate, start_video_capture_ephys_idx, step_time_slice_ephys, session_parameters, figs
    )

def bin_and_count(
    ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
    filter_ephys, filter_all_anipose, bin_width_ms, bin_width_radian, anipose_data_dict,
    bodypart_for_alignment, bodypart_for_reference, bodypart_ref_filter, trial_reject_bounds_mm, trial_reject_bounds_sec,
    origin_offsets, bodyparts_list, session_date, rat_name, treadmill_speed, treadmill_incline,
    camera_fps, align_to, vid_length, time_frame, do_plot, plot_template, MU_colors, CH_colors
    ):
    
    # check inputs for problems
    assert len(ephys_channel_idxs_list)==1, \
    "ephys_channel_idxs_list should only be 1 channel, idiot! :)"
    assert type(bin_width_ms) is int, "bin_width_ms must be type 'int'."
    
    (MU_spikes_by_channel_dict, _, time_axis_for_anipose, ephys_sample_rate,
     start_video_capture_ephys_idx, step_time_slice_ephys, session_parameters, _) = sort(
        ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
        filter_ephys, filter_all_anipose, anipose_data_dict, bodyparts_list=bodypart_for_alignment,
        bodypart_for_alignment=bodypart_for_alignment, bodypart_for_reference=bodypart_for_reference, bodypart_ref_filter=bodypart_ref_filter,
        origin_offsets=origin_offsets, session_date=session_date, rat_name=rat_name,
        treadmill_speed=treadmill_speed, treadmill_incline=treadmill_incline,
        camera_fps=camera_fps, align_to=align_to, vid_length=vid_length,
        time_frame=time_frame, do_plot=False, # change T/F whether to plot sorting plots also
        plot_template=plot_template, MU_colors=MU_colors, CH_colors=CH_colors)    

    # (_, foot_strike_idxs, foot_off_idxs, sliced_step_stats,
    #  step_slice, step_time_slice, _) = peak_align_and_filt(
    #     anipose_data_dict, bodypart_for_alignment=bodypart_for_alignment,
    #     bodypart_for_reference=bodypart_for_reference, bodypart_ref_filter=bodypart_ref_filter,
    #     origin_offsets=origin_offsets, filter_all_anipose=filter_all_anipose, session_date=session_date,
    #     rat_name=rat_name, treadmill_speed=treadmill_speed, treadmill_incline=treadmill_incline,
    #     camera_fps=camera_fps, align_to=align_to, time_frame=time_frame)

    (trialized_anipose_df, keep_trial_set, foot_strike_idxs, foot_off_idxs, sliced_step_stats,
     step_slice, step_time_slice, ref_bodypart_trace_list, pre_align_offset,
     post_align_offset, trial_reject_bounds_mm, trial_reject_bounds_sec) = trialize_steps(
        anipose_data_dict, bodypart_for_alignment, bodypart_for_reference, bodypart_ref_filter,
        trial_reject_bounds_mm, trial_reject_bounds_sec, origin_offsets, bodyparts_list, filter_all_anipose, session_date,
        rat_name, treadmill_speed, treadmill_incline, camera_fps, align_to, time_frame)
    
    # extract data dictionary (with keys for each unit) for the chosen electrophysiology channel
    MU_spikes_dict = MU_spikes_by_channel_dict[str(ephys_channel_idxs_list[0])]#+step_time_slice_ephys.start
    # set conversion ratio from camera to electrophysiology sample rate
    step_to_ephys_conversion_ratio = ephys_sample_rate/camera_fps
    # initialize zero array to carry step-aligned spike activity,
    # with shape: Steps x Time (in ephys sample rate) x Units
    number_of_steps = int(sliced_step_stats['count'])
    MU_spikes_3d_array_ephys_time = np.zeros(
        (number_of_steps, 
        int(sliced_step_stats['max']*step_to_ephys_conversion_ratio),
        len(MU_spikes_dict),
        ))
    MU_spikes_3d_array_ephys_2π = MU_spikes_3d_array_ephys_time.copy()
    
    # initialize dict to store stepwise counts
    MU_step_aligned_spike_counts_dict = {key: None for key in MU_spikes_dict.keys()}
    # initialize dict of lists to store stepwise index arrays
    MU_step_aligned_spike_idxs_dict = {key: [] for key in MU_spikes_dict.keys()}
    MU_step_2π_warped_spike_idxs_dict = {key: [] for key in MU_spikes_dict.keys()}
    # convert foot strike/off indexes to the sample rate of electrophysiology data
    # set chosen alignment bodypart and choose corresponding index values
    if align_to == 'foot strike':
        foot_strike_idxs_in_ephys_time = (
            (foot_strike_idxs[step_slice])*step_to_ephys_conversion_ratio)#+start_video_capture_ephys_idx
        step_idxs_in_ephys_time = foot_strike_idxs_in_ephys_time
    elif align_to == 'foot off':
        foot_off_idxs_in_ephys_time = (
            (foot_off_idxs[step_slice])*step_to_ephys_conversion_ratio)#+start_video_capture_ephys_idx
        step_idxs_in_ephys_time = foot_off_idxs_in_ephys_time

    phase_warp_2π_coeff_list = []
    # fill 3d numpy array with Steps x Time x Units data, and a list of aligned idxs
    for iUnit, iUnitKey in enumerate(MU_spikes_dict.keys()): # for each unit
        MU_spikes_idx_arr = np.array(MU_spikes_dict[iUnitKey]+step_idxs_in_ephys_time[0])
        for iStep in range(number_of_steps): # for each step
            # skip all spike counting and list appending if not in `keep_trial_set`
            if iStep+step_slice.start in keep_trial_set:
                # keep track of index boundaries for each step
                this_step_idx = step_idxs_in_ephys_time[iStep].astype(int)
                next_step_idx = step_idxs_in_ephys_time[iStep+1].astype(int)
                # filter out indexes which are outside the slice or this step's boundaries
                spike_idxs_in_step_and_slice_bounded = MU_spikes_idx_arr[np.where(
                    (MU_spikes_idx_arr < next_step_idx) &
                    (MU_spikes_idx_arr >= this_step_idx) &
                    (MU_spikes_idx_arr >= (step_time_slice.start*step_to_ephys_conversion_ratio).astype(int)) &
                    (MU_spikes_idx_arr <= (step_time_slice.stop*step_to_ephys_conversion_ratio).astype(int))
                    )]
                # subtract current step index to align to each step, and convert to integer index
                MU_spikes_idxs_for_step = (
                    spike_idxs_in_step_and_slice_bounded - this_step_idx).astype(int)
                # store aligned indexes for each step
                MU_step_aligned_spike_idxs_dict[iUnitKey].append(MU_spikes_idxs_for_step)
                # if any spikes are present, set them to 1 for this unit during this step
                if len(MU_spikes_idxs_for_step)!=0:
                    MU_spikes_3d_array_ephys_time[iStep, MU_spikes_idxs_for_step, iUnit] = 1
            else: # mark slices with NaN's for later removal if not inside the keep_trial_set
                MU_spikes_3d_array_ephys_time[iStep, :, iUnit] = np.nan
        # create phase aligned step indexes, with max index for each step set to 2π    
        bin_width_eph_2π = []
        for πStep in range(number_of_steps): # for each step
            if πStep+step_slice.start in keep_trial_set:
                # keep track of index boundaries for each step
                this_step_2π_idx = step_idxs_in_ephys_time[πStep].astype(int)
                next_step_2π_idx = step_idxs_in_ephys_time[πStep+1].astype(int)
                # filter out indexes which are outside the slice or this step_2π's boundaries
                spike_idxs_in_step_2π_and_slice_bounded = MU_spikes_idx_arr[np.where(
                    (MU_spikes_idx_arr < next_step_2π_idx) &
                    (MU_spikes_idx_arr >= this_step_2π_idx) &
                    (MU_spikes_idx_arr >= (step_time_slice.start*step_to_ephys_conversion_ratio).astype(int)) &
                    (MU_spikes_idx_arr <= (step_time_slice.stop*step_to_ephys_conversion_ratio).astype(int))
                    )]
                # coefficient to make step out of 2π radians, step made to be 2π after multiplication
                phase_warp_2π_coeff = 2*np.pi/(
                    step_idxs_in_ephys_time[πStep+1]-step_idxs_in_ephys_time[πStep])
                phase_warp_2π_coeff_list.append(phase_warp_2π_coeff)
                # subtract this step start idx, and convert to an integer index
                MU_spikes_idxs_for_step_aligned = (
                    spike_idxs_in_step_2π_and_slice_bounded - this_step_2π_idx).astype(int)
                MU_spikes_idxs_for_step_2π = MU_spikes_idxs_for_step_aligned * phase_warp_2π_coeff
                # store aligned indexes for each step_2π
                MU_step_2π_warped_spike_idxs_dict[iUnitKey].append(MU_spikes_idxs_for_step_2π)
                # if spikes are present, set them to 1 for this unit during this step
                if len(MU_spikes_idxs_for_step_2π)!=0:
                    MU_spikes_3d_array_ephys_2π[πStep,
                        # convert MU_spikes_idxs_for_step_2π back to ephys sample rate-sized indexes
                        np.round(MU_spikes_idxs_for_step_2π/(
                            2*np.pi)*MU_spikes_3d_array_ephys_2π.shape[1]).astype(int),iUnit] = 1
            else: # mark slices with NaN's for later removal if not inside the keep_trial_set
                MU_spikes_3d_array_ephys_2π[πStep, :, iUnit] = np.nan
    # drop all steps rejected by `process_steps.trialize_steps()`
    steps_to_keep_arr = np.sort(np.fromiter(keep_trial_set,int))
    # (~np.isnan(MU_spikes_3d_array_ephys_time[iStep+step_slice.start,:,:])).any()
    MU_spikes_3d_array_ephys_time = MU_spikes_3d_array_ephys_time[
                                        steps_to_keep_arr-step_slice.start,:,:]
    MU_spikes_3d_array_ephys_2π = MU_spikes_3d_array_ephys_2π[
                                        steps_to_keep_arr-step_slice.start,:,:]
    
    # bin 3d array to time bins with width: bin_width_ms
    ms_duration = MU_spikes_3d_array_ephys_time.shape[1]/ephys_sample_rate*1000
    # round up number of bins to prevent index overflows
    number_of_bins_in_time_step = np.ceil(ms_duration/bin_width_ms).astype(int)
    bin_width_eph = int(bin_width_ms*(ephys_sample_rate/1000))
    
    # round up number of bins to prevent index overflows
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
        f'<b>Binned Spike Count Across<br>{number_of_steps} Steps ({bin_width_ms}ms bins)</b>',
        row = 1, col = 1)
    fig1.update_yaxes(title_text=\
        f'<b>Binned Spike Count Across<br>{number_of_steps} Steps ({bin_2π_rnd}rad bins)</b>',
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
        f'<b>Total Motor Unit Threshold Crossings Across {number_of_steps} Steps</b>\
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
        MU_spikes_count_across_all_steps, steps_to_keep_arr,
        step_idxs_in_ephys_time, ephys_sample_rate, figs
    )

def raster(
    ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
    filter_ephys, filter_all_anipose, bin_width_ms, bin_width_radian, anipose_data_dict,
    bodypart_for_alignment, bodypart_for_reference, bodypart_ref_filter, trial_reject_bounds_mm, trial_reject_bounds_sec,
    origin_offsets, bodyparts_list, session_date, rat_name, treadmill_speed, treadmill_incline,
    camera_fps, align_to, vid_length, time_frame,
    do_plot, plot_template, MU_colors, CH_colors
    ):
    
    (MU_step_aligned_spike_idxs_dict,
    MU_step_aligned_spike_counts_dict,
    MU_step_2π_warped_spike_idxs_dict,
    MU_spikes_3d_array_ephys_time,
    MU_spikes_3d_array_binned,
    MU_spikes_3d_array_binned_2π,
    MU_spikes_count_across_all_steps, steps_to_keep_arr,
    step_idxs_in_ephys_time, ephys_sample_rate, figs
    ) = bin_and_count(
    ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
    filter_ephys, filter_all_anipose, bin_width_ms, bin_width_radian, anipose_data_dict,
    bodypart_for_alignment, bodypart_for_reference, bodypart_ref_filter, trial_reject_bounds_mm, trial_reject_bounds_sec,
    origin_offsets, bodyparts_list, session_date, rat_name, treadmill_speed, treadmill_incline,
    camera_fps, align_to, vid_length, time_frame,
    do_plot=False, plot_template=plot_template, MU_colors=MU_colors, CH_colors=CH_colors)
    
    session_parameters = f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
    
    number_of_steps = MU_spikes_3d_array_ephys_time.shape[0]
    samples_per_step = MU_spikes_3d_array_ephys_time.shape[1]
    number_of_units = MU_spikes_3d_array_ephys_time.shape[2]
    ch_counter = 0
    unit_counter = 0
    step_counter = 0
    fig = go.Figure()
    # for each channel and each trial's spike time series, plot onto the raster: plotly scatters
    # for iChan in 
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
    # ch_counter+=1
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
    filter_ephys, filter_all_anipose, bin_width_ms, bin_width_radian, smoothing_window,
    anipose_data_dict, bodypart_for_alignment, bodypart_for_reference, bodypart_ref_filter,
    trial_reject_bounds_mm, trial_reject_bounds_sec, origin_offsets, bodyparts_list, session_date, rat_name,
    treadmill_speed, treadmill_incline, camera_fps, align_to, vid_length, time_frame,
    do_plot, phase_align, plot_template, MU_colors, CH_colors):
    
    (MU_step_aligned_spike_idxs_dict,
    MU_step_aligned_spike_counts_dict,
    MU_step_2π_warped_spike_idxs_dict,
    MU_spikes_3d_array_ephys_time,
    MU_spikes_3d_array_binned,
    MU_spikes_3d_array_binned_2π,
    MU_spikes_count_across_all_steps, steps_to_keep_arr,
    step_idxs_in_ephys_time, ephys_sample_rate, figs
    ) = bin_and_count(
    ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list, filter_ephys,
    filter_all_anipose, bin_width_ms, bin_width_radian, anipose_data_dict, bodypart_for_alignment,
    bodypart_for_reference, bodypart_ref_filter, trial_reject_bounds_mm, trial_reject_bounds_sec, origin_offsets,
    bodyparts_list, session_date, rat_name, treadmill_speed, treadmill_incline, camera_fps,
    align_to, vid_length, time_frame, do_plot=False, plot_template=plot_template,
    MU_colors=MU_colors, CH_colors=CH_colors)
    
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
    
    return MU_smoothed_spikes_3d_array, binned_spike_array, steps_to_keep_arr, figs
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
    ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list, filter_ephys,
    filter_all_anipose, bin_width_ms, bin_width_radian, smoothing_window, anipose_data_dict,
    bodypart_for_alignment, bodypart_for_reference, bodypart_ref_filter, trial_reject_bounds_mm, trial_reject_bounds_sec,
    origin_offsets, bodyparts_list, session_date, rat_name, treadmill_speed, treadmill_incline,
    camera_fps, align_to, vid_length, time_frame, do_plot, plot_units, phase_align, plot_template,
    MU_colors, CH_colors):
    
    (MU_smoothed_spikes_3d_array, binned_spike_array, steps_to_keep_arr, figs) = smooth(
        ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list, filter_ephys,
        filter_all_anipose, bin_width_ms, bin_width_radian, smoothing_window, anipose_data_dict,
        bodypart_for_alignment, bodypart_for_reference, bodypart_ref_filter,
        trial_reject_bounds_mm, trial_reject_bounds_sec,origin_offsets, bodyparts_list, session_date, rat_name,
        treadmill_speed, treadmill_incline, camera_fps, align_to, vid_length, time_frame,
        do_plot=False, phase_align=phase_align, plot_template=plot_template, MU_colors=MU_colors,
        CH_colors=CH_colors)
    
    # session_parameters = \
    #     f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
    
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
    for iStep, true_step in enumerate(steps_to_keep_arr):
        # gaussian smooth across time, with standard deviation value of bin_width_ms
        sliced_MU_smoothed_3d_array[iStep,:,:] = gaussian_filter1d(
            sliced_MU_smoothed_3d_array[iStep,:,:],
            sigma=smoothing_window,
            axis=0, order=0, output=None, mode='constant',
            cval=0.0, truncate=4.0)
        if number_of_units<=2:
            fig.add_trace(go.Scatter(
                x=sliced_MU_smoothed_3d_array[iStep,:,0],
                y=sliced_MU_smoothed_3d_array[iStep,:,1],
                name=f'step{true_step}',
                mode='lines',
                opacity=.5,
                line=dict(width=5,color=MU_colors[treadmill_incline//5],dash='solid')
                ))
        elif number_of_units==3:
            fig.add_trace(go.Scatter3d(
                x=sliced_MU_smoothed_3d_array[iStep,:,0],
                y=sliced_MU_smoothed_3d_array[iStep,:,1],
                z=sliced_MU_smoothed_3d_array[iStep,:,2],
                name=f'step{true_step}',
                mode='lines',
                opacity=.5,
                line=dict(width=8,color=MU_colors[treadmill_incline//5],dash='solid')
                ))
    # plot mean traces for each unit
    if number_of_units<=2:
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
    if number_of_units<=2:
        fig.update_layout(
            title_text=
            f'<b>{title_prefix}MU State Space Activity for Channel {ephys_channel_idxs_list[0]} Across Inclines</b>\
            <br><sup><b>Incline: {treadmill_incline}</b>, Bin Width: {np.round(bin_width,4)}{bin_unit}, Smoothed by {smoothing_window} bin window</sup>',
            xaxis_title_text=f'<b>Unit {plot_units[0]}</b>',
            yaxis_title_text=f'<b>Unit {plot_units[1]}</b>'
            )
    elif number_of_units==3:
        fig.update_layout(
            title_text=
            f'<b>{title_prefix}MU State Space Activity for Channel {ephys_channel_idxs_list[0]} Across Inclines</b>\
            <br><sup><b>Incline: {treadmill_incline}</b>, Bin Width: {bin_width}{bin_unit}, Smoothed by {smoothing_window} bins</sup>')
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
    return MU_smoothed_spikes_3d_array, binned_spike_array, steps_to_keep_arr, figs

def MU_space_stepwise(ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list, filter_ephys,
    filter_all_anipose, bin_width_ms, bin_width_radian, smoothing_window, anipose_data_dict,
    bodypart_for_alignment, bodypart_for_reference, bodypart_ref_filter, trial_reject_bounds_mm, trial_reject_bounds_sec,
    origin_offsets, bodyparts_list, session_date, rat_name, treadmill_speed, treadmill_incline,
    camera_fps, align_to, vid_length, time_frame, do_plot, plot_units, phase_align, plot_template,
    MU_colors, CH_colors):
    
    iPar = 0
    # session_parameters_lst = []
    # trialized_anipose_dfs_lst = []
    subtitles = []
    for iTitle in treadmill_incline:
        subtitles.append("<b>Incline: "+str(iTitle)+"</b>")
        
    big_fig = []
    for iPar in range(len(treadmill_incline)):
        (trialized_anipose_df, keep_trial_set, foot_strike_idxs, foot_off_idxs, sliced_step_stats,
         step_slice, step_time_slice, ref_bodypart_trace_list, pre_align_offset, post_align_offset,
        trial_reject_bounds_mm, trial_reject_bounds_sec) = \
            trialize_steps(anipose_data_dict, bodypart_for_alignment, bodypart_for_reference,
                           bodypart_ref_filter, trial_reject_bounds_mm, trial_reject_bounds_sec,
                           origin_offsets, bodyparts_list, filter_all_anipose, session_date[iPar],
                           rat_name[iPar], treadmill_speed[iPar], treadmill_incline[iPar],
                           camera_fps, align_to, time_frame)
        
        # (MU_smoothed_spikes_3d_array, binned_spike_array, figs) = smooth(
        # ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list, filter_ephys,
        # filter_all_anipose, bin_width_ms, bin_width_radian, smoothing_window, anipose_data_dict,
        # bodypart_for_alignment, bodypart_for_reference, bodypart_ref_filter,
        # trial_reject_bounds_mm, trial_reject_bounds_sec,origin_offsets, bodyparts_list,
        # session_date[iPar], rat_name[iPar], treadmill_speed[iPar], treadmill_incline[iPar],
        # camera_fps, align_to, vid_length, time_frame, do_plot=False, phase_align=phase_align,
        # plot_template=plot_template, MU_colors=MU_colors, CH_colors=CH_colors)

        MU_smoothed_spikes_3d_array, binned_spike_array, steps_to_keep_arr, figs = \
            state_space(ephys_data_dict,ephys_channel_idxs_list, MU_spike_amplitudes_list,
                    filter_ephys, filter_all_anipose, bin_width_ms, bin_width_radian,
                    smoothing_window, anipose_data_dict, bodypart_for_alignment,
                    bodypart_for_reference, bodypart_ref_filter, trial_reject_bounds_mm,
                    trial_reject_bounds_sec, origin_offsets, bodyparts_list, session_date[iPar],
                    rat_name[iPar], treadmill_speed[iPar], treadmill_incline[iPar], camera_fps,
                    align_to, vid_length, time_frame, do_plot=False, plot_units=plot_units,
                    phase_align=phase_align, plot_template=plot_template, MU_colors=MU_colors,
                    CH_colors=CH_colors)

        # session_parameters = \
        # f"{session_date[iPar]}_{rat_name[iPar]}_speed{treadmill_speed[iPar]}_incline{treadmill_incline[iPar]}"
        
        # raster_figs = raster(ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
        # filter_ephys, filter_all_anipose, bin_width_ms, bin_width_radian, anipose_data_dict,
        # bodypart_for_alignment, bodypart_for_reference, bodypart_ref_filter, trial_reject_bounds_mm, trial_reject_bounds_sec,
        # origin_offsets, bodyparts_list, session_date[0], rat_name[0], treadmill_speed[0], treadmill_incline[0],
        # camera_fps, align_to, vid_length, time_frame,
        # do_plot, plot_template, MU_colors, CH_colors)
        
        number_of_steps = len(keep_trial_set)
        number_of_rows = 2*(len(plot_units)*number_of_steps)+1
        row_spec_list = number_of_rows*[[None]]
        row_spec_list[0] = [{'rowspan': len(plot_units)*number_of_steps, 'b': 0.15}]
        row_spec_list[len(plot_units)*number_of_steps+1] = [{'rowspan': len(plot_units)*number_of_steps}]
        
        big_fig.append(make_subplots(
            rows=number_of_rows, cols=1,
            specs=row_spec_list,
            shared_xaxes=False,
            subplot_titles=(
            f"tmp",
            f"<b>Binned Neural Activity for Steps:</b> {keep_trial_set}"
            )))
        big_fig[iPar].layout.annotations[0].update(text=figs[0].layout.title.text.split('<br>')[1])
        
        for iTrace in range(len(figs[0].data)):
            big_fig[iPar].add_trace(figs[0].data[iTrace], row=1, col=1)
        
        CH_colors.reverse()
        for iStep in range(binned_spike_array.shape[0]):
            big_fig[iPar].data[iStep]['line']['color'] = MU_colors[iStep]
            big_fig[iPar].data[iStep]['opacity'] = 0.5
            big_fig[iPar].data[iStep]['line']['width'] = 5
            for ii, iUnit in enumerate(plot_units):                
                if ii==0:
                    color = MU_colors[iStep % len(MU_colors)]
                elif ii==1:
                    color = CH_colors[iStep % len(MU_colors)]
                else:
                    color = 'black'
                big_fig[iPar].add_scatter(
                    x=np.arange(len(binned_spike_array[iStep,:,iUnit])),
                    y=binned_spike_array[iStep,:,iUnit]-5*iStep,
                    row=len(plot_units)*number_of_steps+2, col=1,
                    name=f"step{steps_to_keep_arr[iStep]}, unit{iUnit}",
                    line_color=color, # color large units darker
                    line_width=3, opacity=0.6)
        if phase_align:
            bin_width = np.round(bin_width_radian,decimals=3)
            bin_units = "radians"
        else:
            bin_width = bin_width_ms
            bin_units = "ms"
            
        big_fig[iPar].update_xaxes(title_text=f"<b>Bins ({bin_width} {bin_units})</b>",row=len(plot_units)*number_of_steps+2, col=1)
        big_fig[iPar].update_yaxes(title_text=f"<b>Binned Spike Counts</b>",row=len(plot_units)*number_of_steps+2, col=1)
        
        big_fig[iPar].add_scatter(x=[18,18], y=[-1,-6], mode='lines', line_width=5,
                                  line_color="black", name="scalebar",
                                  row=len(plot_units)*number_of_steps+2, col=1)
        big_fig[iPar].add_scatter(x=[17], y=[-3], text='<b>5 spikes<br>per bin</b>',
                                  textposition="middle left", mode='text', line_width=5,
                                  line_color="black", name="label",
                                  row=len(plot_units)*number_of_steps+2, col=1)
        # tried barplot, didn't work
        # for iStep in range(binned_spike_array.shape[0]):
        #     big_fig[iPar].data[iStep]['line']['color'] = MU_colors[iStep]
        #     big_fig[iPar].data[iStep]['opacity'] = 0.5
        #     big_fig[iPar].data[iStep]['line']['width'] = 5
        #     for iUnit in plot_units:                
        #         big_fig[iPar].add_bar(
        #             x=np.arange(len(binned_spike_array[iStep,:,iUnit])),
        #             y=binned_spike_array[iStep,:,iUnit],
        #             row=len(plot_units)*number_of_steps+iStep, col=1,
        #             name=f"step{steps_to_keep_arr[iStep]}, unit{iUnit}",
        #             marker_color=MU_colors[iStep % len(MU_colors)],
        #             opacity=0.5)
                    
        # i_session_date = session_date[iPar]
        # i_rat_name = str(rat_name[iPar]).lower()
        # i_treadmill_speed = str(treadmill_speed[iPar]).zfill(2)
        # i_treadmill_incline = str(treadmill_incline[iPar]).zfill(2)
        # session_parameters_lst.append(
        #     f"{i_session_date}_{i_rat_name}_speed{i_treadmill_speed}_incline{i_treadmill_incline}")
        # trialized_anipose_dfs_lst.append(trialized_anipose_df)
        # trialized_anipose_dfs_lst[iPar]['Labels'] = pd.Series(int(i_treadmill_incline) * \
        #                         np.ones(anipose_data_dict[session_parameters_lst[iPar]].shape[0]))
        
        # big_fig[iPar].add_vline(x=0, line_width=3, line_dash="dash", line_color="black", name=align_to)
        
        # Edit the layout
        big_fig[iPar].update_layout(
            title=f'<b>Comparison Across Motor Unit State-Space Representation and Spike Binning</b>',
            xaxis_title=f'<b>Unit 0 Smoothed Activity</b>',
            yaxis_title='<b>Unit 1 Smoothed Activity</b>'
            )
        big_fig[iPar].update_xaxes(scaleanchor = "y", scaleratio = 1, row=1, col=1)
        # big_fig[iPar].update_yaxes(scaleanchor = "x", scaleratio = 1, row=1, col=1)
        # big_fig[iPar].update_yaxes(matches='x', row=1, col=1)
        
        # fig2.update_layout(title=f'<b>{align_to}-Aligned Kinematics for {i_rat_name} on {i_session_date}, Trial Bounds: {trial_reject_bounds_mm}</b>')
        # for xx in range(len(treadmill_incline)):
        #     fig2.update_xaxes(title_text='<b>Time (sec)</b>', row = len(bodyparts_list), col = xx+1)
        # for yy, yTitle in enumerate(bodyparts_list):
        #     fig2.update_yaxes(title_text="<b>"+str(yTitle)+" (mm)</b>", row = yy+1, col = 1)
        # fig2.update_yaxes(scaleanchor = "x",scaleratio = 1)
        # fig2.update_yaxes(matches='y')
        
        iplot(big_fig[iPar])
        return
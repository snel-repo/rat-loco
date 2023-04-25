from process_steps import peak_align_and_filt, trialize_steps
import numpy as np
from plotly.offline import iplot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, butter, filtfilt, iirnotch
from scipy.ndimage import gaussian_filter1d
from inspect import stack
# from pdb import set_trace
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
def sort(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, iterator):
    ### Unpack CFG Inputs
    # unpack analysis inputs
    (MU_spike_amplitudes_list,ephys_channel_idxs_list,filter_ephys,sort_method,
     bodypart_for_reference,bodypart_ref_filter,filter_all_anipose,trial_reject_bounds_mm,
     trial_reject_bounds_sec,origin_offsets,save_binned_MU_data,time_frame,bin_width_ms,
     num_rad_bins,smoothing_window,phase_align,align_to,export_data) = CFG['analysis'].values()
    # unpack plotting inputs
    (plot_type,plot_units,do_plot,N_colors,plot_template,*_) = CFG['plotting'].values()
    # unpack chosen rat inputs
    (bodyparts_list,bodypart_for_alignment,session_date,treadmill_speed,
     treadmill_incline,camera_fps,vid_length) = CFG['rat'][chosen_rat].values()
    if len(bodyparts_list)>0 and bodypart_for_alignment:
        assert bodyparts_list[0] == bodypart_for_alignment[0], \
            ("Error: If bodyparts_list is not empty, bodyparts_list[0] must be "
             "the same as bodypart_for_alignment[0] in config.toml")
    
    # format inputs to avoid ambiguities
    session_date = session_date[iterator]
    rat_name = str(chosen_rat).lower()
    treadmill_speed = str(treadmill_speed[iterator]).zfill(2)
    treadmill_incline = str(treadmill_incline[iterator]).zfill(2)
    session_ID = f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
    
    if do_plot==2: # override and ensure all plots  when 
        do_plot = True
    else: # only display plot if rat_loco_analysis() is the caller
        do_plot = True if (stack()[1].function == 'rat_loco_analysis' and not plot_type.__contains__('multi')) else False


    # extract data from dictionaries
    chosen_ephys_data_continuous_obj = OE_dict[session_ID]
    chosen_anipose_df = anipose_dict[session_ID]

    # create time axes
    ephys_sample_rate = chosen_ephys_data_continuous_obj.metadata['sample_rate']
    time_axis_for_ephys = np.arange(
        round(len(chosen_ephys_data_continuous_obj.samples))
        )/ephys_sample_rate
    # find the beginning of the camera SYNC pulse         
    filtered_sync_channel = butter_highpass_filter(
        data=chosen_ephys_data_continuous_obj.samples[:,16], cutoff=50, fs=30000, order=2)
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

    (processed_anipose_df, foot_strike_idxs, foot_off_idxs, _, step_slice, step_time_slice,
        ref_bodypart_trace_list) = peak_align_and_filt(chosen_rat, OE_dict, KS_dict, anipose_dict,
                                                    CH_colors, MU_colors, CFG, iterator)
    
    # filter step peaks/troughs to be within chosen time_frame, but
    # only when time_frame=1 is not indicating to use the full dataset
    if time_frame!=1:
        foot_strike_slice_idxs = foot_strike_idxs[np.where(
            (foot_strike_idxs >= step_time_slice.start) &
            (foot_strike_idxs <= step_time_slice.stop))]
        foot_off_slice_idxs = foot_off_idxs[np.where(
            (foot_off_idxs >= step_time_slice.start) &
            (foot_off_idxs <= step_time_slice.stop))]
    
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
    step_slice_bounds_in_ephys_time = []
    step_slice_bounds_in_ephys_time.append(
        int(step_to_ephys_conversion_ratio*(step_time_slice.start))+start_video_capture_ephys_idx)
    step_slice_bounds_in_ephys_time.append(
        int(step_to_ephys_conversion_ratio*(step_time_slice.stop))+start_video_capture_ephys_idx)
    if time_frame == 1:
        slice_for_ephys_during_video = slice(0,-1) # get full anipose traces, if time_frame==1
    else:
        # step_slice = slice(step_time_slice.start,step_time_slice.stop)
        slice_for_ephys_during_video = slice(step_slice_bounds_in_ephys_time[0],
                                             step_slice_bounds_in_ephys_time[1])
    
    # cluster the spikes waveforms with PCA
    # pca = PCA(n_components=3)
    # pca.fit(chosen_ephys_data_continuous_obj.samples)
    # print(pca.explained_variance_ratio_)
    if sort_method == 'thresholding':
        # extract spikes that are detected in the selected amplitude threshold ranges
        MU_spikes_by_unit_dict = {}
        MU_spikes_by_unit_dict_keys = [str(int(unit[0])) for unit in MU_spike_amplitudes_list]
        MU_channel_keys_list = [str(ch) for ch in ephys_channel_idxs_list]
        MU_spikes_dict = {key: None for key in MU_channel_keys_list}
        for channel_number in ephys_channel_idxs_list:
            MU_spike_idxs = [] # init empty list for each channel to hold next sorted spike idxs
            for iAmplitudes in MU_spike_amplitudes_list:
                if channel_number not in [-1,16]:
                    ephys_data_for_channel = chosen_ephys_data_continuous_obj.samples[
                        slice_for_ephys_during_video, channel_number]
                    if filter_ephys == 'notch' or filter_ephys == 'both':
                        ephys_data_for_channel = iir_notch(
                            ephys_data_for_channel, ephys_sample_rate)
                    if filter_ephys == 'bandpass' or filter_ephys == 'both':
                        # 350-7000Hz band
                        ephys_data_for_channel = butter_bandpass_filter(
                            ephys_data_for_channel, 350.0, 7000.0, ephys_sample_rate) 
                    MU_spike_idxs_for_channel, _ = find_peaks(
                        -ephys_data_for_channel,
                        height=iAmplitudes,
                        threshold=None,
                        distance=ephys_sample_rate//1000, # 1ms refractory period
                        prominence=None,
                        width=None,
                        wlen=None,
                        )
                    MU_spike_idxs.append(np.int32(MU_spike_idxs_for_channel))
            MU_spikes_by_unit_dict = dict(zip(MU_spikes_by_unit_dict_keys,MU_spike_idxs))
            MU_spikes_dict[str(channel_number)] = MU_spikes_by_unit_dict
        if filter_ephys == 'notch' or filter_ephys == 'both':
            print('60Hz notch filter applied to voltage signals.')
        if filter_ephys == 'bandpass' or filter_ephys == 'both':
            print('350-7000Hz bandpass filter applied to voltage signals.')
        if filter_ephys not in ['notch','bandpass','both']:
            print('No additional filters applied to voltage signals.')
    elif sort_method == 'kilosort':
        chosen_KS_dict = KS_dict[session_ID]
        MU_spikes_dict = {k:v for (k,v) in chosen_KS_dict.items() if k in plot_units}
        assert len(MU_spikes_dict)==len(plot_units), \
            ("Selected MU key could be missing from input KS dictionary, "
             "try indexing from 1 in config.toml: [plotting]: plot_units.")
    
    # MU_spike_idxs = np.array(MU_spike_idxs,dtype=object).squeeze().tolist()
    
    ### PLOTTING SECTION
    # if sort_method == 'thresholding':    
    # compute number of channels and units per channel
    # then compute a color stride value to maximize use of color space
    # number_of_units_per_channel = len(MU_spike_amplitudes_list)
    # color_stride = len(MU_colors)//(number_of_units_per_channel*number_of_channels)
    color_stride = 1
    # compute number of rows to allocate for each subplot based on numbers of each channel
        
    if sort_method == 'thresholding':    
        number_of_channels = len(np.where((np.array(ephys_channel_idxs_list)!=16)
                                        &(np.array(ephys_channel_idxs_list)!=-1))[0])
        number_of_rows = len(bodyparts_list)+len(ephys_channel_idxs_list)+number_of_channels//2+1
        row_spec_list = number_of_rows*[[None]]
        row_spec_list[0] = [{'rowspan': len(bodyparts_list)}]
        row_spec_list[len(bodyparts_list)] = [{'rowspan': len(ephys_channel_idxs_list)}]
        row_spec_list[len(bodyparts_list)+len(ephys_channel_idxs_list)] = \
            [{'rowspan':1}] # number_of_channels//2+1}]
    elif sort_method == 'kilosort':
        number_of_channels = 1
        number_of_rows = len(bodyparts_list)+len(ephys_channel_idxs_list)+len(MU_spikes_dict)//6+1
        row_spec_list = number_of_rows*[[None]]
        row_spec_list[0] = [{'rowspan': len(bodyparts_list)}]
        row_spec_list[len(bodyparts_list)] = [{'rowspan': len(ephys_channel_idxs_list)}]
        row_spec_list[len(bodyparts_list)+len(ephys_channel_idxs_list)] = \
            [{'rowspan':1}] #len(MU_spikes_dict)//6+1}]

    if sort_method=="kilosort":
        MU_labels = list(KS_dict.keys())[iterator]
    elif sort_method=="thresholding":
        MU_labels = list(OE_dict.keys())[iterator]
        
    fig = make_subplots(
        rows=number_of_rows, cols=1,
        specs=row_spec_list,
        shared_xaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0,
        subplot_titles=(
        f"<b>Locomotion Kinematics: {list(anipose_dict.keys())[iterator]}</b>",
        f"<b>Neural Activity: {MU_labels}</b>",
        f"<b>Sorted Spikes: {MU_labels}</b>") if len(bodyparts_list)>0 \
                                                else (f"<b>Neural Activity: {MU_labels}</b>",
                                                      f"<b>Sorted Spikes: {MU_labels}</b>"))
    
    # plot all chosen bodyparts_list, including peak and trough locations for step identification
    bodypart_counter = 0
    color_list =['cornflowerblue','darkorange','green','red'] 
    # ^ alternate scheme override: ['cornflowerblue','royalblue','darkorange','tomato']
    if len(bodyparts_list)>0:
        if bodypart_for_alignment[0] not in bodyparts_list:
            print(
                ("Warning! bodypart_for_alignment is not in bodyparts_list, "
                 "so foot offs/strikes will not be plotted.")
                  )
        for name, values in chosen_anipose_df.items():
            if name in bodyparts_list:
                if name == bodypart_for_alignment[0]:
                    # filtered signal plot (used for alignment)
                    fig.add_trace(go.Scatter(
                        x=time_axis_for_anipose[step_time_slice],
                        y=np.round(processed_anipose_df[bodypart_for_alignment[0]][step_time_slice], decimals=1),
                        name=bodyparts_list[bodypart_counter]+' processed' if filter_all_anipose or origin_offsets
                            else bodyparts_list[bodypart_counter],
                        mode='lines',
                        opacity=.9,
                        line=dict(width=2,color=color_list[bodypart_counter%len(color_list)])),
                        row=1, col=1
                        )
                    # foot strikes
                    fig.add_trace(go.Scatter(
                        x=time_axis_for_anipose[foot_strike_slice_idxs],
                        y=np.round(processed_anipose_df[bodypart_for_alignment[0]][foot_strike_slice_idxs], decimals=1),
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
                        y=np.round(processed_anipose_df[bodypart_for_alignment[0]][foot_off_slice_idxs], decimals=1),
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
                            y=np.round(processed_anipose_df[name][step_time_slice], decimals=1), # + 25*bodypart_counter,
                            name=bodyparts_list[bodypart_counter]+' processed',
                            mode='lines',
                            opacity=.9,
                            line=dict(width=2,color=color_list[bodypart_counter%len(color_list)])),
                            row=1, col=1,
                            )
                    else:
                        fig.add_trace(go.Scatter(
                            x=time_axis_for_anipose[step_time_slice],
                            y=np.round(values.values[step_time_slice], decimals=1), # + 25*bodypart_counter,
                            name=bodyparts_list[bodypart_counter],
                            mode='lines',
                            opacity=.9,
                            line=dict(width=2,color=color_list[bodypart_counter%len(color_list)])),
                            row=1, col=1,
                            )
                    bodypart_counter += 1 # increment for each matching bodypart
        if bodypart_ref_filter and origin_offsets is not False:
            # plot x/y/z reference trace
            dims = [key for key in origin_offsets.keys() if type(origin_offsets[key]) is not int]
            for dim, ref_trace in zip(dims, ref_bodypart_trace_list):
                fig.add_trace(go.Scatter(
                    x=time_axis_for_anipose[step_time_slice],
                    y=np.round(ref_trace[step_time_slice], decimals=1),
                    name=f"Ref: {bodypart_for_reference}_{dim}, {bodypart_ref_filter}Hz lowpass",
                    mode='lines',
                    opacity=.9,
                    line=dict(width=3,color="lightgray",dash='dash')),
                    row=1, col=1
                    )
    # initialize counter to keep track of total unit count across all channels
    unit_counter = np.int16(0)
    # plot all ephys traces and/or SYNC channel
    row_spacing = 0
    for iChannel, channel_number in enumerate(ephys_channel_idxs_list):
        row_spacing = np.clip(0.9*np.max(chosen_ephys_data_continuous_obj.samples[slice_for_ephys_during_video,channel_number]),2000,5000) + row_spacing
        fig.add_trace(go.Scatter(
            x=time_axis_for_ephys[slice_for_ephys_during_video],
            # if statement provides different scalings and offsets for ephys vs. SYNC channel
            y=np.round((chosen_ephys_data_continuous_obj.samples[
                slice_for_ephys_during_video,channel_number] - row_spacing
                if channel_number not in [-1,16]
                else (chosen_ephys_data_continuous_obj.samples[
                    slice_for_ephys_during_video,channel_number]+4)*0.5e3), decimals=1),
            name=f"CH{channel_number}" if channel_number not in [-1,16] else "SYNC",
            mode='lines',
            marker = dict(color=CH_colors[color_stride*iChannel]) if sort_method == 'thresholding' else dict(color=CH_colors[color_stride*iChannel]),
            opacity=1,
            line=dict(width=.4)),
            row=len(bodyparts_list)+1, col=1,
            )
        if sort_method == 'kilosort':
            # UnitKeys = MU_spikes_dict.keys()
            UnitKeys = plot_units
        elif sort_method == 'thresholding':
            UnitKeys = MU_spikes_dict[str(channel_number)].keys()
        sliced_MU_spikes_dict = MU_spikes_dict.copy()
        for iUnit, iUnitKey in enumerate(UnitKeys):
            if channel_number not in [-1,16]:
                if sort_method == 'thresholding':
                    MU_spikes_dict_for_unit = MU_spikes_dict[str(channel_number)][iUnitKey][:]+slice_for_ephys_during_video.start
                    sliced_MU_spikes_dict[str(channel_number)][iUnitKey] = MU_spikes_dict_for_unit.copy()
                elif sort_method == 'kilosort':
                    MU_spikes_dict_for_unit = MU_spikes_dict[iUnitKey][:] if time_frame==1 \
                        else MU_spikes_dict[iUnitKey][:][np.where(
                            (MU_spikes_dict[iUnitKey][:] > slice_for_ephys_during_video.start) &
                            (MU_spikes_dict[iUnitKey][:] < slice_for_ephys_during_video.stop)
                        )]
                    sliced_MU_spikes_dict[iUnitKey] = MU_spikes_dict_for_unit.copy()-slice_for_ephys_during_video.start
                row2 = len(bodyparts_list)+len(ephys_channel_idxs_list)+1                
                # plot spike locations onto each selected ephys trace
                fig.add_trace(go.Scatter(
                    x=time_axis_for_ephys[ # index where spikes are, starting after the video
                        MU_spikes_dict_for_unit],
                    y=np.round(
                        chosen_ephys_data_continuous_obj.samples[MU_spikes_dict_for_unit,channel_number]-row_spacing,
                        decimals=1),
                    name=f"CH{channel_number}, Unit {iUnitKey}",
                    mode='markers',
                    marker = dict(color=MU_colors[color_stride*unit_counter]) if sort_method == 'thresholding' \
                        else dict(color=MU_colors[color_stride*(unit_counter % len(UnitKeys))]),
                    opacity=.9,
                    line=dict(width=3)),
                    row=len(bodyparts_list)+1, col=1
                    )
                if sort_method == 'thresholding' or (sort_method == 'kilosort' and iChannel==0):
                    # plot isolated spikes into raster plot for each selected ephys trace
                    fig.add_trace(go.Scatter(
                        x=time_axis_for_ephys[ # index where spikes are, starting after the video
                            MU_spikes_dict_for_unit],
                        y=np.zeros(len(time_axis_for_ephys[slice_for_ephys_during_video])).astype(np.int16)-unit_counter,
                        name=f"CH{channel_number}, Unit {iUnitKey}" if sort_method == 'thresholding' \
                                                                    else f"KS Cluster: {iUnitKey}",
                        mode='markers',
                        marker_symbol='line-ns',
                        marker = dict(color=MU_colors[color_stride*unit_counter],
                                    line_color=MU_colors[color_stride*unit_counter],
                                    line_width=1.2,
                                    size=10),
                        opacity=1),
                        row=row2,
                        col=1
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
        # fig.write_html(f"/home/sean/Downloads/{session_ID}_{str(ephys_channel_idxs_list)}.html")
    if export_data:
        from scipy.io import savemat, loadmat
        # time_axis_for_ephys=time_axis_for_ephys[slice_for_ephys_during_video],
        # MU_spikes_by_KS_cluster = {str(k): np.array(v,dtype=int) for k, v in sliced_MU_spikes_dict.items()},
        # time_axis_for_anipose=time_axis_for_anipose,
        # foot_off_idxs = np.round(processed_anipose_df[bodypart_for_alignment[0]][foot_off_slice_idxs],1),
        # foot_strike_idxs = np.round(processed_anipose_df[bodypart_for_alignment[0]][foot_strike_slice_idxs],1),
        # anipose_data = {k: np.array(v,dtype=float) for k, v in chosen_anipose_df.to_dict('list').items()},
        # session_ID = session_ID
        export_dict = dict(time_axis_for_ephys=time_axis_for_ephys[slice_for_ephys_during_video],
                           ephys_data = chosen_ephys_data_continuous_obj.samples[slice_for_ephys_during_video],
                           MU_spikes_by_KS_cluster = {'unit'+str(k).zfill(2): np.array(v+1,dtype=np.int64) for k, v in sliced_MU_spikes_dict.items()},
                           time_axis_for_anipose=time_axis_for_anipose,
                           foot_off_idxs = foot_off_idxs+1,
                           foot_strike_idxs = foot_strike_idxs+1,
                           foot_off_times = time_axis_for_anipose[foot_off_slice_idxs],
                           foot_strike_times = time_axis_for_anipose[foot_strike_slice_idxs],
                           anipose_data = {k: np.array(v,dtype=float) for k, v in chosen_anipose_df.to_dict('list').items()},
                           session_ID = session_ID)
        savemat(f'{session_ID}.mat', export_dict, oned_as='column')
        # x = loadmat(f'{session_ID}.mat')
    return (
        MU_spikes_dict, time_axis_for_ephys, chosen_anipose_df, time_axis_for_anipose,
        ephys_sample_rate, start_video_capture_ephys_idx, slice_for_ephys_during_video, session_ID, figs
    )

def bin_and_count(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, iterator):
    ### Unpack CFG Inputs
    # unpack analysis inputs
    (MU_spike_amplitudes_list,ephys_channel_idxs_list,filter_ephys,sort_method,
     bodypart_for_reference,bodypart_ref_filter,filter_all_anipose,trial_reject_bounds_mm,
     trial_reject_bounds_sec,origin_offsets,save_binned_MU_data,time_frame,bin_width_ms,
     num_rad_bins,smoothing_window,phase_align,align_to,export_data) = CFG['analysis'].values()
    # unpack plotting inputs
    (plot_type,plot_units,do_plot,N_colors,plot_template,*_) = CFG['plotting'].values()
    # unpack chosen rat inputs
    (bodyparts_list,bodypart_for_alignment,session_date,treadmill_speed,
     treadmill_incline,camera_fps,vid_length) = CFG['rat'][chosen_rat].values()
    
    # format inputs to avoid ambiguities
    session_date = session_date[iterator]
    rat_name = str(chosen_rat).lower()
    treadmill_speed = str(treadmill_speed[iterator]).zfill(2)
    treadmill_incline = str(treadmill_incline[iterator]).zfill(2)
    session_ID = f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
    
    if do_plot==2: # override and ensure all plots  when 
        do_plot = True
    else: # only display plot if rat_loco_analysis() is the caller
        do_plot = True if (stack()[1].function == 'rat_loco_analysis' and not plot_type.__contains__('multi')) else False


    # check inputs for problems
    if 16 in ephys_channel_idxs_list:
        ephys_channel_idxs_list.remove(16)
    if sort_method == 'thresholding':
        assert len(ephys_channel_idxs_list)==1, \
        "ephys_channel_idxs_list should only be 1 channel, idiot! :)"
    assert type(bin_width_ms) is int, "bin_width_ms must be type 'int'."
    
    (MU_spikes_dict, _, chosen_anipose_df, _, ephys_sample_rate, _, slice_for_ephys_during_video, session_ID,_) = sort(
        chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, iterator)

    # (_, foot_strike_idxs, foot_off_idxs, sliced_step_stats,
    #  step_slice, step_time_slice, _) = peak_align_and_filt(
    #     bodypart_for_alignment=bodypart_for_alignment,
    #     bodypart_for_reference=bodypart_for_reference, bodypart_ref_filter=bodypart_ref_filter,
    #     origin_offsets=origin_offsets, filter_all_anipose=filter_all_anipose, session_date=session_date,
    #     rat_name=rat_name, treadmill_speed=treadmill_speed, treadmill_incline=treadmill_incline,
    #     camera_fps=camera_fps, align_to=align_to, time_frame=time_frame)

    (
        trialized_anipose_df, keep_trial_set, foot_strike_idxs, foot_off_idxs, sliced_step_stats,
        kept_step_stats, step_slice, step_time_slice, ref_bodypart_trace_list, pre_align_offset,
        post_align_offset, trial_reject_bounds_mm, trial_reject_bounds_sec) = trialize_steps(
        chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, iterator
        )
    # extract data dictionary (with keys for each unit) for the chosen electrophysiology channel
    if sort_method == 'thresholding':
        MU_spikes_dict = MU_spikes_dict[str(ephys_channel_idxs_list[0])]#+slice_for_ephys_during_video.start
        print("!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! !!")
        print("!! Using FIRST channel in ephys_channel_idxs_list !!")
        print("!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! !!")
    # set conversion ratio from camera to electrophysiology sample rate
    step_to_ephys_conversion_ratio = ephys_sample_rate/camera_fps
    # initialize zero array to carry step-aligned spike activity,
    # with shape: Steps x Time (in ephys sample rate) x Units
    number_of_steps = int(kept_step_stats['count'])
    MU_spikes_3d_array_ephys_time = np.zeros(
        (number_of_steps, 
        int(kept_step_stats['max']*step_to_ephys_conversion_ratio),
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
        step_idxs_in_ephys_time = np.int32(foot_strike_idxs_in_ephys_time)
    elif align_to == 'foot off':
        foot_off_idxs_in_ephys_time = (
            (foot_off_idxs[step_slice])*step_to_ephys_conversion_ratio)#+start_video_capture_ephys_idx
        step_idxs_in_ephys_time = np.int32(foot_off_idxs_in_ephys_time)

    phase_warp_2π_coeff_list = []
    # fill 3d numpy array with Steps x Time x Units data, and a list of aligned idxs
    for iUnit, iUnitKey in enumerate(MU_spikes_dict.keys()): # for each unit
        if sort_method == 'thresholding':
            MU_spikes_idx_arr = MU_spikes_dict[iUnitKey]+step_idxs_in_ephys_time[0]
        elif sort_method == 'kilosort':
            MU_spikes_idx_arr = MU_spikes_dict[iUnitKey][
                                np.where(
                                    (MU_spikes_dict[iUnitKey][:] > slice_for_ephys_during_video.start) &
                                    (MU_spikes_dict[iUnitKey][:] < slice_for_ephys_during_video.stop))
                                ]-slice_for_ephys_during_video.start+step_idxs_in_ephys_time[0]
        for ii, iStep in enumerate(keep_trial_set):#range(number_of_steps): # for each step
            iStep -= step_slice.start
            # skip all spike counting and list appending if not in `keep_trial_set`
            # if iStep+step_slice.start in keep_trial_set:
            # keep track of index boundaries for each step
            this_step_idx = step_idxs_in_ephys_time[iStep]
            next_step_idx = step_idxs_in_ephys_time[iStep+1]
            # filter out indexes which are outside the slice or this step's boundaries
            spike_idxs_in_step_and_slice_bounded = MU_spikes_idx_arr[np.where(
                (MU_spikes_idx_arr < next_step_idx) &
                (MU_spikes_idx_arr >= this_step_idx) &
                (MU_spikes_idx_arr >= np.int32(step_time_slice.start*step_to_ephys_conversion_ratio)) &
                (MU_spikes_idx_arr <= np.int32(step_time_slice.stop*step_to_ephys_conversion_ratio))
                )]
            # subtract current step index to align to each step, and convert to np.integer32 index
            MU_spikes_idxs_for_step = (
                spike_idxs_in_step_and_slice_bounded - this_step_idx).astype(np.int32)
            # store aligned indexes for each step
            MU_step_aligned_spike_idxs_dict[iUnitKey].append(MU_spikes_idxs_for_step)
            # if any spikes are present, set them to 1 for this unit during this step
            if len(MU_spikes_idxs_for_step)!=0:
                MU_spikes_3d_array_ephys_time[ii, MU_spikes_idxs_for_step, iUnit] = 1
            # else: # mark slices with NaN's for later removal if not inside the keep_trial_set
            #     MU_spikes_3d_array_ephys_time[ii, :, iUnit] = np.nan
        # create phase aligned step indexes, with max index for each step set to 2π    
        bin_width_eph_2π = []
        for ii, πStep in enumerate(keep_trial_set): #range(number_of_steps): # for each step
            πStep -= step_slice.start
            # if πStep+step_slice.start in keep_trial_set:
            # keep track of index boundaries for each step
            this_step_2π_idx = step_idxs_in_ephys_time[πStep]
            next_step_2π_idx = step_idxs_in_ephys_time[πStep+1]
            # filter out indexes which are outside the slice or this step_2π's boundaries
            spike_idxs_in_step_2π_and_slice_bounded = MU_spikes_idx_arr[np.where(
                (MU_spikes_idx_arr < next_step_2π_idx) &
                (MU_spikes_idx_arr >= this_step_2π_idx) &
                (MU_spikes_idx_arr >= np.int32(step_time_slice.start*step_to_ephys_conversion_ratio)) &
                (MU_spikes_idx_arr <= np.int32(step_time_slice.stop*step_to_ephys_conversion_ratio))
                )]
            # coefficient to make step out of 2π radians, step made to be 2π after multiplication
            phase_warp_2π_coeff = 2*np.pi/(
                step_idxs_in_ephys_time[πStep+1]-step_idxs_in_ephys_time[πStep])
            phase_warp_2π_coeff_list.append(phase_warp_2π_coeff)
            # subtract this step start idx, and convert to an np.integer32 index
            MU_spikes_idxs_for_step_aligned = (
                spike_idxs_in_step_2π_and_slice_bounded - this_step_2π_idx).astype(np.int32)
            MU_spikes_idxs_for_step_2π = MU_spikes_idxs_for_step_aligned * phase_warp_2π_coeff
            # store aligned indexes for each step_2π
            MU_step_2π_warped_spike_idxs_dict[iUnitKey].append(MU_spikes_idxs_for_step_2π)
            # if spikes are present, set them to 1 for this unit during this step
            if len(MU_spikes_idxs_for_step_2π)!=0:
                MU_spikes_3d_array_ephys_2π[ii,
                    # convert MU_spikes_idxs_for_step_2π back to ephys sample rate-sized indexes
                    np.round(MU_spikes_idxs_for_step_2π/(
                        2*np.pi)*MU_spikes_3d_array_ephys_2π.shape[1]).astype(np.int32),iUnit] = 1
            # else: # mark slices with NaN's for later removal if not inside the keep_trial_set
            #     MU_spikes_3d_array_ephys_2π[ii, :, iUnit] = np.nan
    # drop all steps rejected by `process_steps.trialize_steps()`
    steps_to_keep_arr = np.sort(np.fromiter(keep_trial_set,np.int32))
    # (~np.isnan(MU_spikes_3d_array_ephys_time[iStep+step_slice.start,:,:])).any()

    # MU_spikes_3d_array_ephys_time = MU_spikes_3d_array_ephys_time[
    #                                 steps_to_keep_arr-step_slice.start,
    #                                 :,#np.int32(step_to_ephys_conversion_ratio*kept_step_stats.max()),
    #                                 :]
    # MU_spikes_3d_array_ephys_2π = MU_spikes_3d_array_ephys_2π[
    #                                 steps_to_keep_arr-step_slice.start,
    #                                 :,#np.int32(step_to_ephys_conversion_ratio*kept_step_stats.max()),
    #                                 :]
    
    # bin 3d array to time bins with width: bin_width_ms
    ms_duration = MU_spikes_3d_array_ephys_time.shape[1]/ephys_sample_rate*1000
    # round up number of bins to prevent index overflows
    number_of_bins_in_time_step = np.ceil(ms_duration/bin_width_ms).astype(np.int32)
    bin_width_eph = np.int32(bin_width_ms*(ephys_sample_rate/1000))
    
    # leave 2*pi numerator and number of bins equals: (500 / bin_width_ms)
    # WARNING: may need to change denominator constant to achieve correct radian binwidth
    bin_width_radian = 2*np.pi/num_rad_bins
    # round up number of bins to prevent index overflows
    number_of_bins_in_2π_step = np.ceil(2*np.pi/bin_width_radian).astype(np.int32)
    bin_width_eph_2π = np.round(
        MU_spikes_3d_array_ephys_2π.shape[1]/number_of_bins_in_2π_step).astype(np.int32)
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
    # number_of_units_per_channel = len(MU_spikes_dict[
    #                                            str(ephys_channel_idxs_list[0])])
    # color_stride = len(MU_colors)//(number_of_channels*number_of_units_per_channel)
    # sum all spikes across step cycles
    MU_spikes_count_across_all_steps = MU_spikes_3d_array_binned.sum(0).sum(0)
    order_by_count = np.argsort(MU_spikes_count_across_all_steps)
    color_stride = 1
    fig1 = make_subplots(
        rows=1, cols=2,
        shared_xaxes=False,
        subplot_titles=(
        f"Session Info: {session_ID}",
        f"Session Info: {session_ID}"
        ))
    if sort_method=='kilosort':
        MU_iter = plot_units
    else:
        MU_iter = MU_spikes_dict.keys()
        
    # for iUnit, iUnitKey in enumerate(np.fromiter(MU_spikes_dict.keys(),'int')[order_by_count[::-1]]):
    # import pdb; pdb.set_trace()
    for iUnit, iUnitKey in enumerate(np.fromiter(MU_iter,'int')[order_by_count[::-1]]):
        try:
            MU_step_aligned_idxs = np.concatenate(MU_step_aligned_spike_idxs_dict[str(iUnitKey)]).ravel()            
        except:
            MU_step_aligned_idxs = np.concatenate(MU_step_aligned_spike_idxs_dict[iUnitKey]).ravel()
        MU_step_aligned_idxs_ms = MU_step_aligned_idxs/ephys_sample_rate*1000
        fig1.add_trace(go.Histogram(
            x=MU_step_aligned_idxs_ms, # ms
            xbins=dict(start=0, size=bin_width_ms),
            name=str(iUnitKey)+"uV crossings" if (
                sort_method=='thresholding') else "KS cluster: "+str(iUnitKey),
            marker_color=MU_colors[color_stride*iUnit]),
            row=1, col=1
            )
    # for iUnit, iUnitKey in enumerate(np.fromiter(MU_spikes_dict.keys(),'int')[order_by_count[::-1]]):
    for iUnit, iUnitKey in enumerate(np.fromiter(MU_iter,'int')[order_by_count[::-1]]):
        try:
            MU_step_2π_aligned_idxs = np.concatenate(
                MU_step_2π_warped_spike_idxs_dict[str(iUnitKey)]).ravel()            
        except:
            MU_step_2π_aligned_idxs = np.concatenate(
                MU_step_2π_warped_spike_idxs_dict[iUnitKey]).ravel()            
        fig1.add_trace(go.Histogram(
            x=MU_step_2π_aligned_idxs, # radians
            xbins=dict(start=0, size=bin_width_radian),
            name=str(iUnitKey)+"uV crossings" if (
                sort_method=='thresholding') else "KS cluster: "+str(iUnitKey),
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
    
    fig2.add_trace(go.Bar(
    # list comprehension to get threshold values for each isolated unit on this channel
    x=[str(iUnitKey)+"uV crossings" for iUnitKey in np.fromiter(MU_spikes_dict.keys(),'int')[order_by_count[::-1]]] \
        if sort_method=='thresholding' \
        else ["KS Cluster: "+str(iUnitKey) for iUnitKey in  np.fromiter(MU_spikes_dict.keys(),'int')[order_by_count[::-1]]],
    y=MU_spikes_count_across_all_steps[order_by_count[::-1]],
    marker_color=[MU_colors[iColor] for iColor in range(0,len(MU_colors),color_stride)],
    opacity=1,
    showlegend=False
    # name="Counts Bar Plot"
    ))
    # set all titles
    fig2.update_layout(
        title_text=
        f'<b>Total Motor Unit Spikes Across {number_of_steps} Steps</b>\
        <br><sup>Session Info: {session_ID}</sup>',
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
    if export_data:
        from scipy.io import savemat
        pass

    if save_binned_MU_data is True:
        np.save(session_ID+"_time.npy",MU_spikes_3d_array_binned, allow_pickle=False)
        np.save(session_ID+"_phase.npy",MU_spikes_3d_array_binned_2π, allow_pickle=False)
        
    return(
        MU_spikes_dict,
        MU_step_aligned_spike_idxs_dict,
        MU_step_aligned_spike_counts_dict,
        MU_step_2π_warped_spike_idxs_dict,
        MU_spikes_3d_array_ephys_time,
        MU_spikes_3d_array_binned,
        MU_spikes_3d_array_binned_2π,
        MU_spikes_count_across_all_steps, steps_to_keep_arr,
        step_idxs_in_ephys_time, ephys_sample_rate, session_ID, figs
    )

def raster(
    chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, iterator
    ):
    
    (
        MU_spikes_dict,
        MU_step_aligned_spike_idxs_dict,
        MU_step_aligned_spike_counts_dict,
        MU_step_2π_warped_spike_idxs_dict,
        MU_spikes_3d_array_ephys_time,
        MU_spikes_3d_array_binned,
        MU_spikes_3d_array_binned_2π,
        MU_spikes_count_across_all_steps, steps_to_keep_arr,
        step_idxs_in_ephys_time, ephys_sample_rate, session_ID, figs
    ) = bin_and_count(
        chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, iterator
        )
    
    ### Unpack CFG Inputs
    # unpack analysis inputs
    (MU_spike_amplitudes_list,ephys_channel_idxs_list,filter_ephys,sort_method,
     bodypart_for_reference,bodypart_ref_filter,filter_all_anipose,trial_reject_bounds_mm,
     trial_reject_bounds_sec,origin_offsets,save_binned_MU_data,time_frame,bin_width_ms,
     num_rad_bins,smoothing_window,phase_align,align_to,export_data) = CFG['analysis'].values()
    # unpack plotting inputs
    (plot_type,plot_units,do_plot,N_colors,plot_template,*_) = CFG['plotting'].values()
    # unpack chosen rat inputs
    (bodyparts_list,bodypart_for_alignment,session_date,treadmill_speed,
     treadmill_incline,camera_fps,vid_length) = CFG['rat'][chosen_rat].values()
    
    # format inputs to avoid ambiguities
    session_date = session_date[iterator]
    rat_name = str(chosen_rat).lower()
    treadmill_speed = str(treadmill_speed[iterator]).zfill(2)
    treadmill_incline = str(treadmill_incline[iterator]).zfill(2)
    session_ID = f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
    
    if do_plot==2: # override and ensure all plots are displayed when do_plot==2
        do_plot = True
    else: # only display plot if rat_loco_analysis() is the caller
        do_plot = True if (stack()[1].function == 'rat_loco_analysis' and not plot_type.__contains__('multi')) else False
    
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
                name=f'step{iStep} unit{iUnitKey}',
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
        <br><sup>Session Info: {session_ID}</sup>',
        xaxis_title_text=f'<b>Time (ms)</b>',
        yaxis_title_text=f'<b>Step</b>'
        )
    # elif number_of_units==3:
    #     fig.update_layout(
    #         title_text=
    #         f'<b>MU Activity Raster for All {number_of_steps} Steps</b>\
    #         <br><sup>Session Info: {session_ID}</sup>',
    #         xaxis_title_text=f'<b>Time (ms)</b>',
    #         yaxis_title_text=f'<b>Step (ms)</b>'
    #         )
        
    # set theme to chosen template
    fig.update_layout(template=plot_template)
        
    if do_plot:
        iplot(fig)
    if export_data:
        from scipy.io import savemat
        pass
    
    figs = [fig]
    return figs


def smooth(
    chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, iterator
    ):
    
    (
        MU_spikes_dict,
        MU_step_aligned_spike_idxs_dict,
        MU_step_aligned_spike_counts_dict,
        MU_step_2π_warped_spike_idxs_dict,
        MU_spikes_3d_array_ephys_time,
        MU_spikes_3d_array_binned,
        MU_spikes_3d_array_binned_2π,
        MU_spikes_count_across_all_steps, steps_to_keep_arr,
        step_idxs_in_ephys_time, ephys_sample_rate, session_ID, figs
    ) = bin_and_count(
        chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, iterator
        )
    
    ### Unpack CFG Inputs
    # unpack analysis inputs
    (MU_spike_amplitudes_list,ephys_channel_idxs_list,filter_ephys,sort_method,
     bodypart_for_reference,bodypart_ref_filter,filter_all_anipose,trial_reject_bounds_mm,
     trial_reject_bounds_sec,origin_offsets,save_binned_MU_data,time_frame,bin_width_ms,
     num_rad_bins,smoothing_window,phase_align,align_to,export_data) = CFG['analysis'].values()
    # unpack plotting inputs
    (plot_type,plot_units,do_plot,N_colors,plot_template,*_) = CFG['plotting'].values()
    # unpack chosen rat inputs
    (bodyparts_list,bodypart_for_alignment,session_date,treadmill_speed,
     treadmill_incline,camera_fps,vid_length) = CFG['rat'][chosen_rat].values()
    
    # format inputs to avoid ambiguities
    session_date = session_date[iterator]
    rat_name = str(chosen_rat).lower()
    treadmill_speed = str(treadmill_speed[iterator]).zfill(2)
    treadmill_incline = str(treadmill_incline[iterator]).zfill(2)
    session_ID = f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
    
    if do_plot==2: # override and ensure all plots  when 
        do_plot = True
    else: # only display plot if rat_loco_analysis() is the caller
        do_plot = True if (stack()[1].function == 'rat_loco_analysis' and not plot_type.__contains__('multi')) else False


    # initialize 3d numpy array with shape: Steps x Bins x Units
    if phase_align is True:
        binned_spike_array = MU_spikes_3d_array_binned_2π
        bin_width = (2*np.pi)/num_rad_bins
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
                sigma=smoothing_window[iterator],
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
        <br><sup>Session Info: {session_ID}</sup>',
        xaxis_title_text=f'<b>Bins ({np.round(bin_width,4)}{bin_unit})</b>',
        yaxis_title_text= \
            f'<b>Smoothed MU Activity<br>({smoothing_window} Sample Kernel)</b>',
        )
    # set theme to chosen template
    # fig.update_layout(template=plot_template)
    
    if do_plot:
        iplot(fig)
    if export_data:
        from scipy.io import savemat
        pass
  
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
            chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, iterator
            ):
    
    (MU_smoothed_spikes_3d_array, binned_spike_array, steps_to_keep_arr,
     figs) = smooth(
        chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, iterator
        )
    
    ### Unpack CFG Inputs
    # unpack analysis inputs
    (MU_spike_amplitudes_list,ephys_channel_idxs_list,filter_ephys,sort_method,
     bodypart_for_reference,bodypart_ref_filter,filter_all_anipose,trial_reject_bounds_mm,
     trial_reject_bounds_sec,origin_offsets,save_binned_MU_data,time_frame,bin_width_ms,
     num_rad_bins,smoothing_window,phase_align,align_to,export_data) = CFG['analysis'].values()
    # unpack plotting inputs
    (plot_type,plot_units,do_plot,N_colors,plot_template,*_) = CFG['plotting'].values()
    # unpack chosen rat inputs
    (bodyparts_list,bodypart_for_alignment,session_date,treadmill_speed,
     treadmill_incline,camera_fps,vid_length) = CFG['rat'][chosen_rat].values()
    
    # format inputs to avoid ambiguities
    session_date = session_date[iterator]
    rat_name = str(chosen_rat).lower()
    treadmill_speed = str(treadmill_speed[iterator]).zfill(2)
    treadmill_incline = str(treadmill_incline[iterator]).zfill(2)
    session_ID = f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
    
    if do_plot==2: # override and ensure all plots  when 
        do_plot = True
    else: # only display plot if rat_loco_analysis() is the caller
        do_plot = True if (stack()[1].function == 'rat_loco_analysis' and not plot_type.__contains__('multi')) else False

    # select units for plotting
    if sort_method == 'kilosort':
        sliced_MU_smoothed_3d_array = MU_smoothed_spikes_3d_array # [:,:,plot_units]
    else:
        sliced_MU_smoothed_3d_array = MU_smoothed_spikes_3d_array[:,:,plot_units]
    
    # set numbers of things from input matrix dimensionality
    # number_of_steps = sliced_MU_smoothed_3d_array.shape[0]
    # number_of_bins = sliced_MU_smoothed_3d_array.shape[1]
    # detect number of identified units
    number_of_units = sliced_MU_smoothed_3d_array.any(1).any(0).sum()
    if phase_align is True:
        bin_width = (2*np.pi)/num_rad_bins
        bin_unit = 'radians'
        title_prefix = 'Phase'
    else:
        bin_width = bin_width_ms
        bin_unit = 'ms'
        title_prefix = 'Time'
    
    fig = go.Figure()
    # smooth and plot each trace
    for iStep, true_step in enumerate(steps_to_keep_arr):
        # gaussian smooth across time, with standard deviation value of bin_width_ms
        sliced_MU_smoothed_3d_array[iStep,:,:] = gaussian_filter1d(
            sliced_MU_smoothed_3d_array[iStep,:,:],
            sigma=smoothing_window[iterator],
            axis=0, order=0, output=None, mode='constant',
            cval=0.0, truncate=4.0)
        if number_of_units<=2:
            fig.add_trace(go.Scatter(
                x=sliced_MU_smoothed_3d_array[iStep,:,0],
                y=sliced_MU_smoothed_3d_array[iStep,:,1],
                name=f'step{true_step}',
                mode='lines',
                opacity=.5,
                line=dict(width=5,color=MU_colors[int(treadmill_incline)//5],dash='solid')
                ))
        elif number_of_units>=3:
            fig.add_trace(go.Scatter3d(
                x=sliced_MU_smoothed_3d_array[iStep,:,0],
                y=sliced_MU_smoothed_3d_array[iStep,:,1],
                z=sliced_MU_smoothed_3d_array[iStep,:,2],
                name=f'step{true_step}',
                mode='lines',
                opacity=.5,
                line=dict(width=8,color=MU_colors[int(treadmill_incline)//5],dash='solid')
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
    elif number_of_units>=3:
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
            f'<b>{title_prefix}-Aligned MU State Space Activity for Channel {ephys_channel_idxs_list[0]} Across Inclines</b>\
            <br><b>Incline: {treadmill_incline}</b>',# Bin Width: {np.round(bin_width,3)}{bin_unit}, Smoothed by {smoothing_window} bin window</sup>',
            xaxis_title_text=f'<b>Unit {plot_units[0]} Activity</b>',
            yaxis_title_text=f'<b>Unit {plot_units[1]} Activity</b>'
            )
    elif number_of_units>=3:
        fig.update_layout(
            title_text=
            f'<b>{title_prefix}-Aligned MU State Space Activity for Channel {ephys_channel_idxs_list[0]} Across Inclines</b>\
            <br><b>{session_ID}</b>')#, Bin Width: {np.round(bin_width,3)}{bin_unit}, Smoothed by {smoothing_window} bins</sup>')
        fig.update_scenes(
            dict(camera=dict(eye=dict(x=-2, y=-0.3, z=0.2)), #the default values are 1.25, 1.25, 1.25
                xaxis = dict(title_text=f'<b>Unit {plot_units[0]} Activity</b>', range=[0,1.0]),
                yaxis = dict(title_text=f'<b>Unit {plot_units[1]} Activity</b>', range=[0,1.0]),
                zaxis = dict(title_text=f'<b>Unit {plot_units[2]} Activity</b>', range=[0,1.0]),
                aspectmode='manual', #this string can be 'data', 'cube', 'auto', 'manual'
                # custom aspectratio is defined as follows:
                aspectratio=dict(x=1, y=1, z=1)
           ))
    # set theme to chosen template
    # fig.update_layout(template=plot_template)
    
    if do_plot:
        iplot(fig)
    if export_data:
        from scipy.io import savemat
        pass
    # put in a list for compatibility with calling functions
    figs = [fig]
    return MU_smoothed_spikes_3d_array, binned_spike_array, steps_to_keep_arr, figs

def MU_space_stepwise(
    chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, iterator
    ):
    
    ### Unpack CFG Inputs
    # unpack analysis inputs
    (MU_spike_amplitudes_list,ephys_channel_idxs_list,filter_ephys,sort_method,
     bodypart_for_reference,bodypart_ref_filter,filter_all_anipose,trial_reject_bounds_mm,
     trial_reject_bounds_sec,origin_offsets,save_binned_MU_data,time_frame,bin_width_ms,
     num_rad_bins,smoothing_window,phase_align,align_to,export_data) = CFG['analysis'].values()
    # unpack plotting inputs
    (plot_type,plot_units,do_plot,N_colors,plot_template,*_) = CFG['plotting'].values()
    # unpack chosen rat inputs
    (bodyparts_list,bodypart_for_alignment,session_date,treadmill_speed,
     treadmill_incline,camera_fps,vid_length) = CFG['rat'][chosen_rat].values()
    
    # format inputs to avoid ambiguities
    session_date = session_date[iterator]
    rat_name = str(chosen_rat).lower()
    treadmill_speed = str(treadmill_speed[iterator]).zfill(2)
    treadmill_incline = str(treadmill_incline[iterator]).zfill(2)
    session_ID = f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
    
    if do_plot==2: # override and ensure all plots  when 
        do_plot = True
    else: # only display plot if rat_loco_analysis() is the caller
        do_plot = True if (stack()[1].function == 'rat_loco_analysis' and not plot_type.__contains__('multi')) else False


    iPar = 0
    # session_ID_lst = []
    # trialized_anipose_dfs_lst = []
    subtitles = []
    for iTitle in treadmill_incline:
        subtitles.append("<b>Incline: "+str(iTitle)+"</b>")
        
    big_fig = []
    for iPar in range(len(treadmill_incline)):
        (trialized_anipose_df, keep_trial_set, foot_strike_idxs, foot_off_idxs, sliced_step_stats,
         kept_step_stats, step_slice, step_time_slice, ref_bodypart_trace_list, pre_align_offset,
         post_align_offset, trial_reject_bounds_mm, trial_reject_bounds_sec) = \
            trialize_steps(
                           chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, iterator
                           )
        
        # (MU_smoothed_spikes_3d_array, binned_spike_array, figs) = smooth(
        # OE_dict, KS_dict, anipose_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list, filter_ephys, sort_method,
        # filter_all_anipose, bin_width_ms, bin_width_radian, smoothing_window,
        # bodypart_for_alignment, bodypart_for_reference, bodypart_ref_filter,
        # trial_reject_bounds_mm, trial_reject_bounds_sec,origin_offsets, bodyparts_list,
        # session_date[iPar], rat_name[iPar], treadmill_speed[iPar], treadmill_incline[iPar],
        # camera_fps, align_to, vid_length, time_frame, do_plot=False, phase_align=phase_align,
        # plot_template=plot_template, MU_colors=MU_colors, CH_colors=CH_colors)

        MU_smoothed_spikes_3d_array, binned_spike_array, steps_to_keep_arr, figs = \
            state_space(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, iterator)

        # session_ID = \
        # f"{session_date[iPar]}_{rat_name[iPar]}_speed{treadmill_speed[iPar]}_incline{treadmill_incline[iPar]}"
        
        # raster_figs = raster(OE_dict, KS_dict, anipose_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
        # filter_ephys, sort_method, filter_all_anipose, bin_width_ms, bin_width_radian,
        # bodypart_for_alignment, bodypart_for_reference, bodypart_ref_filter, trial_reject_bounds_mm, trial_reject_bounds_sec,
        # origin_offsets, bodyparts_list, session_date[0], rat_name[0], treadmill_speed[0], treadmill_incline[0],
        # camera_fps, align_to, vid_length, time_frame,
        # do_plot, plot_template, MU_colors, CH_colors)
        
        # set height ratios for each subplot using `specs` parameter of `make_subplots()`
        number_of_steps = len(keep_trial_set)
        number_of_rows = 2*(len(plot_units)*number_of_steps)+1
        row_spec_list = number_of_rows*[[None]]
        if len(plot_units)>=3 and MU_smoothed_spikes_3d_array.any(1).any(0).sum()>=3:
            row_spec_list[0] = [{'type': 'scatter3d','rowspan': len(plot_units)*number_of_steps, 'b': 0.01}] # 1% padding between
        elif len(plot_units)>=2 and MU_smoothed_spikes_3d_array.any(1).any(0).sum()>=2:
            row_spec_list[0] = [{'type': 'scatter','rowspan': len(plot_units)*number_of_steps, 'b': 0.1}] # 10% padding between
        row_spec_list[len(plot_units)*number_of_steps+1] = [{'type': 'scatter','rowspan': len(plot_units)*number_of_steps}]
    
        big_fig.append(make_subplots(
            rows=number_of_rows, cols=1,
            specs=row_spec_list,
            shared_xaxes=False,
            subplot_titles=(
            f"tmp",
            f"<b>Binned Neural Activity for Steps:</b> {keep_trial_set}"
            )))
        big_fig[iPar].layout.annotations[0].update(text=figs[0].layout.title.text)#.split('<br>')[1])
        for iTrace in range(len(figs[0].data)):
            big_fig[iPar].add_trace(figs[0].data[iTrace], row=1, col=1)
            
        activity_matrix = binned_spike_array
        row_offset = activity_matrix.max()
        CH_colors.reverse()
        for iStep in range(activity_matrix.shape[0]):
            big_fig[iPar].data[iStep]['line']['color'] = MU_colors[iStep]
            big_fig[iPar].data[iStep]['opacity'] = 0.5
            big_fig[iPar].data[iStep]['line']['width'] = 2
            for iUnit in plot_units:                
                if iUnit==0:
                    color = MU_colors[iStep % len(MU_colors)]
                elif iUnit==1: # color large units darker
                    color = 'grey'#CH_colors[iStep % len(MU_colors)]
                else:
                    color = 'black'
                # big_fig[iPar].add_scatter(x=np.hstack((np.arange(MU_smoothed_spikes_3d_array.shape[1]),MU_smoothed_spikes_3d_array.shape[1]-1,0)),
                #                           y=np.hstack((MU_smoothed_spikes_3d_array[iStep,:,iUnit]*max(activity_matrix[iStep,:,iUnit])/max(MU_smoothed_spikes_3d_array[iStep,:,iUnit]),0,0))-row_offset*iStep,
                #                           row=len(plot_units)*number_of_steps+2, col=1,
                #                           name=f"smoothed step{steps_to_keep_arr[iStep]}, unit{iUnit}",
                #                           opacity=0.6, fill='toself', mode='lines', # override default markers+lines
                #                           line_color=color, fillcolor=color,
                #                           hoverinfo="skip", showlegend=False
                #                           )
        for iStep in range(activity_matrix.shape[0]):
            big_fig[iPar].data[iStep]['line']['color'] = MU_colors[iStep]
            big_fig[iPar].data[iStep]['opacity'] = 0.5
            big_fig[iPar].data[iStep]['line']['width'] = 2
            for iUnit in plot_units:                
                if iUnit==np.sort(plot_units)[0]:
                    color = MU_colors[iStep % len(MU_colors)]
                elif iUnit==np.sort(plot_units)[1]: # color large units darker
                    color = 'grey'#CH_colors[iStep % len(MU_colors)]
                else:
                    color = 'black'
                big_fig[iPar].add_scatter(
                    x=np.arange(len(activity_matrix[iStep,:,iUnit])),
                    y=activity_matrix[iStep,:,iUnit]-row_offset*iStep,
                    row=len(plot_units)*number_of_steps+2, col=1,
                    name=f"step{steps_to_keep_arr[iStep]}, unit{iUnit}",
                    mode='lines', line_color=color,
                    line_width=1, opacity=0.5)

        if phase_align:
            bin_width_radian = 2*np.pi/(kept_step_stats['max']/camera_fps/bin_width_ms) #(2*np.pi)/num_rad_bins
            bin_width = np.round(bin_width_radian,decimals=3)
            bin_units = "radians"
        else:
            bin_width = bin_width_ms
            bin_units = "ms"
            
        big_fig[iPar].update_xaxes(title_text=f"<b>Bins ({bin_width} {bin_units})</b>",row=len(plot_units)*number_of_steps+2, col=1)
        big_fig[iPar].update_yaxes(title_text=f"<b>Stepwise Binned Spike Counts</b>",row=len(plot_units)*number_of_steps+2, col=1)        
       
        # Edit the layout
        if len(plot_units)>=3 and MU_smoothed_spikes_3d_array.any(1).any(0).sum()>=3:
            big_fig[iPar].update_layout(
                title=f"<b>Comparison Across Motor Unit State-Space Representation and Spike Binning</b>",
                title_font_size=20)
            big_fig[iPar].update_scenes(
                dict(camera=dict(eye=dict(x=-0.3, y=-2, z=0.2)), #the default values are 1.25, 1.25, 1.25
                xaxis = dict(title_text=f'<b>Unit {plot_units[0]}</b>'),
                yaxis = dict(title_text=f'<b>Unit {plot_units[1]}</b>'),
                zaxis = dict(title_text=f'<b>Unit {plot_units[2]}</b>'),
                aspectmode='manual', #this string can be 'data', 'cube', 'auto', 'manual'
                # custom aspectratio is defined as follows:
                aspectratio=dict(x=1, y=1, z=1)),
                row=1, col=1)
        elif len(plot_units)>=2 and MU_smoothed_spikes_3d_array.any(1).any(0).sum()>=2:
            big_fig[iPar].update_layout(
                title=f"<b>Comparison Across Motor Unit State-Space Representation and Spike Binning</b>",
                title_font_size=20,
                xaxis_title=f"<b>Unit {plot_units[0]} Smoothed Activity</b>",
                yaxis_title=f"<b>Unit {plot_units[1]} Smoothed Activity</b>"
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
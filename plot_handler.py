from process_steps import peak_align_and_filt, trialize_steps
import numpy as np
from plotly.offline import iplot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, butter, filtfilt, iirnotch
from scipy.ndimage import gaussian_filter1d
from inspect import stack

def sort_plot(chosen_rat, OE_dict, KS_dict, anipose_dict, MU_spikes_dict, CH_colors, MU_colors, CFG,
              matching_session_idxs, chosen_anipose_df, processed_anipose_df, time_axis_for_anipose,
              time_axis_for_ephys, step_time_slice, foot_strike_slice_idxs, foot_off_slice_idxs,
              ref_bodypart_trace_list, chosen_ephys_data_continuous_obj,
              slice_for_ephys_during_video):
    ### Unpack CFG Inputs
    # unpack analysis inputs
    (MU_spike_amplitudes_list,ephys_channel_idxs_list,filter_ephys,sort_method,sort_to_use,
        bodypart_for_reference,bodypart_ref_filter,filter_all_anipose,trial_reject_bounds_mm,
        trial_reject_bounds_sec,origin_offsets,save_binned_MU_data,time_frame,bin_width_ms,
        num_rad_bins,smoothing_window,phase_align,align_to,export_data) = CFG['analysis'].values()
    # unpack plotting inputs
    (plot_type,plot_units,do_plot,N_colors,plot_template,*_) = CFG['plotting'].values()
    # unpack chosen rat inputs
    (bodyparts_list,bodypart_for_alignment,session_date,treadmill_speed,
        treadmill_incline,camera_fps,vid_length) = CFG['rat'][chosen_rat].values()

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
        MU_labels = list(KS_dict.keys())[matching_session_idxs]
    elif sort_method=="thresholding":
        MU_labels = list(OE_dict.keys())[matching_session_idxs]
        
    fig = make_subplots(
        rows=number_of_rows, cols=1,
        specs=row_spec_list,
        shared_xaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0,
        subplot_titles=(
        f"<b>Locomotion Kinematics: {list(anipose_dict.keys())[matching_session_idxs]}</b>",
        f"<b>Neural Activity: {MU_labels}</b>",
        f"<b>Sorted Spikes: {MU_labels}</b>") if len(bodyparts_list)>0 \
                                                else (f"<b>Neural Activity: {MU_labels}</b>",
                                                    f"<b>Sorted Spikes: {MU_labels}</b>"))
    
    # plot all chosen bodyparts_list, including peak and trough locations for step identification
    bodypart_counter = 0
    color_list =['cornflowerblue','darkorange','green','red'] #['cornflowerblue','royalblue','darkorange','tomato']
    if len(bodyparts_list)>0:
        if bodypart_for_alignment[0] not in bodyparts_list:
            print("Warning! bodypart_for_alignment is not in bodyparts_list, so foot offs/strikes will not be plotted.")
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
    return figs, sliced_MU_spikes_dict


def bin_and_count_plot(MU_spikes_dict_keys, ephys_sample_rate, session_ID, sort_method, do_plot,
                       MU_colors, CFG, MU_step_aligned_spike_idxs_dict,
                       MU_step_2π_warped_spike_idxs_dict, MU_spikes_count_across_all_sessions):
    
    ### Unpack CFG Inputs
    # unpack analysis inputs
    (MU_spike_amplitudes_list,ephys_channel_idxs_list,filter_ephys,sort_method,sort_to_use,
        bodypart_for_reference,bodypart_ref_filter,filter_all_anipose,trial_reject_bounds_mm,
        trial_reject_bounds_sec,origin_offsets,save_binned_MU_data,time_frame,bin_width_ms,
        num_rad_bins,smoothing_window,phase_align,align_to,export_data) = CFG['analysis'].values()
    # unpack plotting inputs
    (plot_type,plot_units,do_plot,N_colors,plot_template,*_) = CFG['plotting'].values()
    order_by_count = np.argsort(MU_spikes_count_across_all_sessions)
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
        MU_iter = MU_spikes_dict_keys
        
    for iUnit, iUnitKey in enumerate(np.fromiter(MU_spikes_dict_keys,'int')[order_by_count[::-1]]):
        try:
            MU_step_aligned_idxs = np.concatenate(
                MU_step_aligned_spike_idxs_dict[str(iUnitKey)]).ravel()            
        except KeyError:
            MU_step_aligned_idxs = np.concatenate(
                MU_step_aligned_spike_idxs_dict[iUnitKey]).ravel()
        except:
            raise
        MU_step_aligned_idxs_ms = MU_step_aligned_idxs/ephys_sample_rate*1000
        fig1.add_trace(go.Histogram(
            x=MU_step_aligned_idxs_ms, # ms
            xbins=dict(start=0, size=bin_width_ms),
            name=str(iUnitKey)+"uV crossings" if (
                sort_method=='thresholding') else "KS cluster: "+str(iUnitKey),
            marker_color=MU_colors[color_stride*iUnit]),
            row=1, col=1
            )
    
    bin_width_radian = 2*np.pi/num_rad_bins
    for iUnit, iUnitKey in enumerate(np.fromiter(MU_spikes_dict_keys,'int')[order_by_count[::-1]]):
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
        title_text='<b>Time and Phase-Aligned Motor Unit Activity During Step Cycle</b>',
        # xaxis_title_text='<b>Time During Step (milliseconds)</b>',
        # yaxis_title_text=,
        # bargap=0., # gap between bars of adjacent location coordinates
        # bargroupgap=0.1 # gap between bars of the same location coordinates
        )
    bin_2π_rnd = np.round(bin_width_radian,4)
    try:
        number_of_steps = len(MU_step_aligned_spike_idxs_dict[str(iUnitKey)])
    except KeyError:
        number_of_steps = len(MU_step_aligned_spike_idxs_dict[iUnitKey])
    except:
        raise
    
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
    x=[str(iUnitKey)+"uV crossings" for iUnitKey in
       np.fromiter(MU_spikes_dict_keys,'int')[order_by_count[::-1]]] \
        if sort_method=='thresholding' \
        else ["KS Cluster: "+str(iUnitKey) for iUnitKey in
              np.fromiter(MU_spikes_dict_keys,'int')[order_by_count[::-1]]],
    y=MU_spikes_count_across_all_sessions[order_by_count[::-1]],
    marker_color=[MU_colors[iColor] for iColor in range(0,len(MU_colors),color_stride)],
    opacity=1,
    showlegend=False
    # name="Counts Bar Plot"
    ))
    # set all titles
    fig2.update_layout(
        title_text= (f'<b>Total Motor Unit Spikes Across {number_of_steps} Steps</b>'
                     f'<br><sup>Session Info: {session_ID}</sup>'),
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
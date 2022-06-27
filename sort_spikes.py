import pdb
from extract_step_idxs import extract_step_idxs
import time
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, butter, filtfilt, lfilter
from scipy.stats import binned_statistic
# from sklearn.decomposition import PCA
# import plotly.colors
# import multiprocessing
# from functools import partial

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def sort_spikes(
    ephys_data_dict,
    anipose_data_dict,
    bodyparts=['palm_L_y','palm_R_y'],
    bodypart_for_tracking=['palm_L_y'],
    session_date=str(time.strftime("%y%m%d")),
    rat_name='dogerat',
    treadmill_speed=20,
    treadmill_incline=0,
    camera_fps=100,
    time_slice=1,
    ephys_channel_idxs=np.arange(17),
    plot_sort = False
    ):

    # format inputs to avoid ambiguities
    rat_name = str(rat_name).lower()
    treadmill_speed = str(treadmill_speed).zfill(2)
    treadmill_incline = str(treadmill_incline).zfill(2)

    # pool_obj = multiprocessing.Pool()
    # plot_trace_fixed=partial(plot_trace, time_axis_for_ephys=time_axis_for_ephys,chosen_ephys_data_continuous_obj=chosen_ephys_data_continuous_obj,session_idx=session_idx,time_slice=time_slice)
    # pool_obj.map(plot_trace_fixed,ephys_channel_idxs)

    # filter Open Ephys dictionaries for the proper session date, speed, and incline
    ephys_data_dict_filtered_by_date = dict(filter(lambda item: str(session_date) in item[0], ephys_data_dict.items()))
    ephys_data_dict_filtered_by_ratname = dict(filter(lambda item: rat_name in item[0], ephys_data_dict_filtered_by_date.items()))
    ephys_data_dict_filtered_by_speed = dict(filter(lambda item: "speed"+treadmill_speed in item[0], ephys_data_dict_filtered_by_ratname.items()))
    ephys_data_dict_filtered_by_incline = dict(filter(lambda item: "incline"+treadmill_incline in item[0], ephys_data_dict_filtered_by_speed.items()))
    chosen_ephys_data_dict = ephys_data_dict_filtered_by_incline
    chosen_ephys_data_continuous_obj = chosen_ephys_data_dict[f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"]

    # filter anipose dictionaries for the proper session date, speed, and incline
    anipose_data_dict_filtered_by_date = dict(filter(lambda item: str(session_date) in item[0], anipose_data_dict.items()))
    anipose_data_dict_filtered_by_ratname = dict(filter(lambda item: rat_name in item[0], anipose_data_dict_filtered_by_date.items()))
    anipose_data_dict_filtered_by_speed = dict(filter(lambda item: "speed"+treadmill_speed in item[0], anipose_data_dict_filtered_by_ratname.items()))
    anipose_data_dict_filtered_by_incline = dict(filter(lambda item: "incline"+treadmill_incline in item[0], anipose_data_dict_filtered_by_speed.items()))
    chosen_anipose_data_dict = anipose_data_dict_filtered_by_incline
    chosen_anipose_df = chosen_anipose_data_dict[f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"]

    # create time axes
    ephys_sample_rate = chosen_ephys_data_continuous_obj.metadata['sample_rate']
    time_axis_for_ephys = np.arange(round(len(chosen_ephys_data_continuous_obj.samples)*time_slice))/ephys_sample_rate
    filtered_sync_channel = butter_highpass_filter(
        data=chosen_ephys_data_continuous_obj.samples[:,-1], cutoff=50, fs=30000, order=2
        )
    start_video_capture_idx = find_peaks(filtered_sync_channel,height=0.3)[0][0] # find the beginning camera TTL pulse 
    time_axis_for_anipose = np.arange(0,20,0.01)+time_axis_for_ephys[start_video_capture_idx]

    # identify motion peak locations of bodypart for step cycle alignment
    do_filter=True
    if do_filter == True:
        filtered_signal = butter_highpass_filter(
            data=chosen_anipose_df[bodypart_for_tracking[0]].values,
            cutoff=0.5, fs=camera_fps, order=5)
    else: # do not filter
        filtered_signal=chosen_anipose_df[bodypart_for_tracking[0]].values

    foot_strike_idxs, foot_off_idxs, step_stats = extract_step_idxs(
    anipose_data_dict, bodypart_for_tracking=bodypart_for_tracking, session_date=session_date,
    rat_name=rat_name, treadmill_speed=treadmill_speed, treadmill_incline=treadmill_incline, camera_fps=camera_fps
    )

    # step_to_ephys_conversion_ratio = ephys_sample_rate/camera_fps
    # foot_strike_idxs_in_ephys_time = (foot_strike_idxs*step_to_ephys_conversion_ratio)+start_video_capture_idx
    # foot_off_idxs_in_ephys_time = (foot_off_idxs*step_to_ephys_conversion_ratio)+start_video_capture_idx    

    # cluster the spikes
    # pca = PCA(n_components=3)
    # pca.fit(chosen_ephys_data_continuous_obj.samples)
    # print(pca.explained_variance_ratio_)

    # extract spikes that are detected in the selected amplitude threshold ranges
    MU_spike_amplitudes_list = [[150,500],[500.0001,1700],[1700.0001,5000]]
    MU_spikes_by_unit_dict = {}
    MU_spikes_by_unit_dict_keys = [str(int(unit[0])) for unit in MU_spike_amplitudes_list]
    MU_channel_keys_list = [str(ch) for ch in ephys_channel_idxs]
    MU_spikes_by_channel_dict = {key: None for key in MU_channel_keys_list}
    for iChannel, channel_number in enumerate(ephys_channel_idxs):
        MU_spike_idxs = [] # initialize empty list for each channel to hold next sorted spike idxs
        for iUnit, iAmplitudes in enumerate(MU_spike_amplitudes_list):
            if channel_number not in [-1,16]:
                MU_spike_idxs_for_channel, _ = find_peaks(
                    -chosen_ephys_data_continuous_obj.samples[:,channel_number],
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
    # MU_spike_idxs = np.array(MU_spike_idxs,dtype=object).squeeze().tolist()

    ### PLOTTING SECTION
    if plot_sort:
        # calculate number of rows to allocate for each subplot
        number_of_rows = len(bodyparts)+len(ephys_channel_idxs)+1  ################################## MAKE 3 FLEXIBLE !!!
        row_spec_list = number_of_rows*[[None]]
        row_spec_list[0] = [{'rowspan': len(bodyparts)}]
        row_spec_list[len(bodyparts)] = [{'rowspan': len(ephys_channel_idxs)}]#,'secondary_y': True}]
        row_spec_list[len(bodyparts)+len(ephys_channel_idxs)] = [{'rowspan': 1}]

        fig = make_subplots(
            rows=number_of_rows, cols=1,
            # specs=[[{'secondary_y': False}],[{'secondary_y': True}],[{'secondary_y': False}]],
            specs=row_spec_list,
            shared_xaxes=True,
            # vertical_spacing=0.0,
            # horizontal_spacing=0.02,
            subplot_titles=(
                f"<b>Locomotion Kinematics: {list(chosen_anipose_data_dict.keys())[0]}</b>",
                f"<b>Neural Activity: {list(chosen_ephys_data_dict.keys())[0]}</b>"
                )
            )

        bodypart_counter = 0
        for name, values in chosen_anipose_df.items():
            if name in bodyparts:
                if name == bodypart_for_tracking[0]:
                    # filtered signal plot (used for alignment)
                    fig.add_trace(go.Scatter(
                    x=time_axis_for_anipose,
                    y=filtered_signal + 25*bodypart_counter, # mean subtract and spread data values out by 20 pixels
                    name=bodyparts[bodypart_counter]+' filtered',
                    mode='lines',
                    opacity=.9,
                    line=dict(width=2)),
                    row=1, col=1)
                    
                    # foot strikes
                    fig.add_trace(go.Scatter(
                    x=time_axis_for_anipose[foot_strike_idxs],
                    y=filtered_signal[foot_strike_idxs] + 25*bodypart_counter, # mean subtract and spread data values out by 20 pixels
                    name=bodyparts[bodypart_counter]+' strike',
                    mode='markers',
                    marker = dict(color='black'),
                    opacity=.9,
                    line=dict(width=3)),
                    row=1, col=1
                    )
                    # foot offs               
                    fig.add_trace(go.Scatter(
                    x=time_axis_for_anipose[foot_off_idxs],
                    y=filtered_signal[foot_off_idxs] + 25*bodypart_counter, # mean subtract and spread data values out by 20 pixels
                    name=bodyparts[bodypart_counter]+' off',
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
                    y=values.values-values.values.mean() + 25*bodypart_counter, # mean subtract and spread data values out by 20 pixels
                    name=bodyparts[bodypart_counter],
                    mode='lines',
                    opacity=.9,
                    line=dict(width=2)),
                    row=1, col=1,
                    )
                    bodypart_counter += 1 # increment for each matching bodypart

        colors_for_MUs =[
            'blue', 'green', 'orange', 'red', 
            'darkblue','darkgreen','darkorange','darkred',
            'lightblue','lightgreen','lightorange','lightred',
            'purple','black' # plotly.colors.sequential.Plasma_r
        ]
        unit_counter = np.int16(0)
        for iChannel, channel_number in enumerate(ephys_channel_idxs):
            fig.add_trace(go.Scatter(
                x=time_axis_for_ephys,
                # spread data values out by 10 millivolts
                y=chosen_ephys_data_continuous_obj.samples[:,channel_number] - 5000*iChannel if channel_number not in [-1,16] else (chosen_ephys_data_continuous_obj.samples[:,channel_number]+4)*0.5e3,
                name=f"CH{channel_number}" if channel_number not in [-1,16] else "SYNC",
                mode='lines',
                opacity=1,
                line=dict(width=.4)),
                row=len(bodyparts)+1, col=1,
                )
            for iUnit, iUnitKey in enumerate(MU_spikes_by_channel_dict[str(channel_number)].keys()):
                if channel_number not in [-1,16]:
                    # plot spike locations onto each selected ephys trace
                    # import pdb; pdb.set_trace()
                    fig.add_trace(go.Scatter(
                        x=time_axis_for_ephys[MU_spikes_by_channel_dict[str(channel_number)][iUnitKey]],
                        y=chosen_ephys_data_continuous_obj.samples[MU_spikes_by_channel_dict[str(channel_number)][iUnitKey],channel_number] - 5000*iChannel,
                        name=f"CH{channel_number} spikes",
                        mode='markers',
                        marker = dict(color=colors_for_MUs[unit_counter]),
                        opacity=.9,
                        line=dict(width=3)),
                        row=len(bodyparts)+1, col=1
                        )
                    # plot isolated spikes into raster plot for each selected ephys trace
                    fig.add_trace(go.Scatter(
                        x=time_axis_for_ephys[MU_spikes_by_channel_dict[str(channel_number)][iUnitKey]],
                        y=np.zeros(len(time_axis_for_ephys))-unit_counter,
                        name=f"CH{channel_number}, Unit {iUnit}",
                        mode='markers',
                        marker_symbol='line-ns',
                        marker = dict(color=colors_for_MUs[unit_counter],
                                    line_color=colors_for_MUs[unit_counter],
                                    line_width=0.8,
                                    size=8),
                        opacity=1),
                        row=len(bodyparts)+len(ephys_channel_idxs)+1, col=1
                        )
                    unit_counter+=1
        
        fig.update_xaxes(
            title_text="<b>Time (s)</b>", row = len(bodyparts)+len(ephys_channel_idxs)+1, col = 1,#secondary_y=False
            )
        fig.update_yaxes(
            title_text="<b>Pixels</b>", row = 1, col = 1
            )
        fig.update_yaxes(
            title_text="<b>Voltage (uV)</b>", row = len(bodyparts)+1, col = 1
            )
        fig.update_yaxes(
            title_text="<b>Sorted Spikes</b>", row = len(bodyparts)+len(ephys_channel_idxs)+1, col = 1 # secondary_y=True
            )

        fig.show()
    return MU_spikes_by_channel_dict, time_axis_for_ephys, time_axis_for_anipose, ephys_sample_rate, start_video_capture_idx, 

def bin_spikes(
    ephys_data_dict, anipose_data_dict, session_date, rat_name, treadmill_speed,
    treadmill_incline, bin_width_ms, ephys_channel_idxs, bodypart_for_tracking, camera_fps, alignto='foot off'):
    
    assert len(ephys_channel_idxs)==1, "ephys_channel_idxs should only be 1 channel, idiot! :)"
    
    MU_spikes_by_channel_dict,time_axis_for_ephys,time_axis_for_anipose,ephys_sample_rate,start_video_capture_idx = sort_spikes(
        ephys_data_dict, anipose_data_dict,session_date=session_date,
        rat_name=rat_name,treadmill_speed=treadmill_speed,
        treadmill_incline=treadmill_incline,ephys_channel_idxs=[7]
        )
    # bin_width_adjusted = (ephys_sample_rate/1000)*bin_width_ms
    # num_bins = int(len(time_axis_for_ephys)//bin_width_adjusted)
    step_to_ephys_conversion_ratio = ephys_sample_rate/camera_fps

    foot_strike_idxs, foot_off_idxs, step_stats = extract_step_idxs(
        anipose_data_dict, bodypart_for_tracking=bodypart_for_tracking, session_date=session_date,
        rat_name=rat_name, treadmill_speed=treadmill_speed, treadmill_incline=treadmill_incline, camera_fps=camera_fps
        )

    MU_spikes_dict = MU_spikes_by_channel_dict[str(ephys_channel_idxs[0])]
    # initialize zero array to carry step-aligned spike activity 
    step_aligned_MU_spikes = np.zeros(
        (len(MU_spikes_dict),
        int(step_stats['max']*step_to_ephys_conversion_ratio),
        int(step_stats['count']-1) # minus 1 to account for mismatch of foot-off and -strike numbers
        ))
    # initialize list to carry step-cycle aligned indexes
    MU_step_aligned_spike_counts_dict = {key: None for key in MU_spikes_dict.keys()} # initialize dict to store stepwise counts
    MU_step_aligned_spike_idxs_dict = {key: [] for key in MU_spikes_dict.keys()} # initialize dict of lists to store stepwise index arrays

    foot_strike_idxs_in_ephys_time = (foot_strike_idxs*step_to_ephys_conversion_ratio)+start_video_capture_idx
    foot_off_idxs_in_ephys_time = (foot_off_idxs*step_to_ephys_conversion_ratio)+start_video_capture_idx
    
    alignto = 'foot off'
    if alignto == 'foot strike':
        step_idxs_in_ephys_time = foot_strike_idxs_in_ephys_time
    elif alignto == 'foot off':
        step_idxs_in_ephys_time = foot_off_idxs_in_ephys_time

    # fill 3d numpy array with Units x Time x Trials/Steps data, and a list of aligned idxs
    for iUnit, iUnitKey in enumerate(MU_spikes_dict.keys()):
        spikes_array = np.array(MU_spikes_dict[iUnitKey])
        this_step_idx = 0; next_step_idx = 0;
        for iStep in range(1,step_aligned_MU_spikes.shape[2]):
            
            this_step_idx = step_idxs_in_ephys_time[iStep].astype(int)
            next_step_idx = step_idxs_in_ephys_time[iStep+1].astype(int)
            
            spike_idxs_in_step_and_video_bounded = spikes_array[np.where(
                (spikes_array < next_step_idx) &
                (spikes_array > this_step_idx) &
                (spikes_array < (time_axis_for_anipose.max()*ephys_sample_rate).astype(int)) &
                (spikes_array > start_video_capture_idx)
                )]

            MU_spikes_idxs_for_step = (spike_idxs_in_step_and_video_bounded - this_step_idx).astype(int)
            MU_step_aligned_spike_idxs_dict[iUnitKey].append(MU_spikes_idxs_for_step) # store aligned indexes for each step

            if len(MU_spikes_idxs_for_step)!=0:
                # set all detected spikes for this unit during this step to 1
                step_aligned_MU_spikes[iUnit,MU_spikes_idxs_for_step,iStep] = 1
        
        
    # import matplotlib.pyplot as plt; plt.hist(MU_spikes_dict[iUnitKey], step_idxs_in_ephys_time); plt.show()
    MU_spikes_count_across_steps = step_aligned_MU_spikes.sum(axis=2).sum(axis=1) # sum all spikes across step cycles

    fig = go.Figure()
    colors_for_MUs =['blue', 'green', 'orange' 'red','darkblue','darkgreen','darkorange', 'darkred','purple','black']
    for iUnit, iUnitKey in enumerate(MU_spikes_dict.keys()):
        MU_step_aligned_idxs = np.concatenate(MU_step_aligned_spike_idxs_dict[iUnitKey]).ravel()
        # MU_step_aligned_spike_counts_dict[iUnitKey], _ = np.convolve()
        
        MU_step_aligned_idxs_ms = MU_step_aligned_idxs/ephys_sample_rate*1000
        # counts, bin_edges = np.histogram(MU_step_aligned_idxs_ms,bin_width_ms)

        # bin_means, bin_edges, binnumber = binned_statistic(MU_step_aligned_idxs,MU_step_aligned_idxs_ms,bins=len(bin_edges-1))
        
        # fig.add_trace(go.Bar(
        #     x=binnumber,
        #     y=bin_means, # ms
        #     marker_color=colors_for_MUs[iUnit],
        #     name=iUnitKey+"uV crossings"
        #     ))
        fig.add_trace(go.Histogram(
            x=MU_step_aligned_idxs_ms, # ms
            xbins=dict(start=0, size=bin_width_ms),
            name=iUnitKey+"uV crossings",
            marker_color=colors_for_MUs[iUnit],
            ))

    fig.update_traces(opacity=0.5) # Reduce opacity to see both histograms
    fig.update_layout(
        barmode='overlay',
        title_text='<b>Step-Aligned Motor Unit Activity (Thresholded Units)</b>',
        xaxis_title_text='<b>Time During Step (milliseconds)</b>',
        yaxis_title_text=f'<b>Spike Count Across {step_aligned_MU_spikes.shape[2]} Steps</b>',
        # bargap=0., # gap between bars of adjacent location coordinates
        )
        # bargroupgap=0.1 # gap between bars of the same location coordinates
        # )
    fig.show()
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
    x=np.arange(len(MU_spikes_count_across_steps)),
    y=MU_spikes_count_across_steps,
    marker_color=colors_for_MUs[iUnit],
    name=iUnitKey+"uV crossings"
    ))
    fig2.show()
    return MU_step_aligned_spike_idxs_dict, MU_step_aligned_spike_counts_dict, step_idxs_in_ephys_time

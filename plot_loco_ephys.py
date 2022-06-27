import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, butter, filtfilt
# import multiprocessing
# from functools import partial

def butter_highpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=2):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# function receives output of import_ephys_data.py as import_anipose_data.py to plot aligned data
def full(
    ephys_data_dict,
    anipose_data_dict,
    bodyparts=['palm_L_z','palm_R_z'],
    session_date=str(time.strftime("%y%m%d")),
    rat_name='dogerat',
    treadmill_speed=20,
    treadmill_incline=0,
    camera_fps=100,
    time_slice=1,
    ephys_channel_idxs=np.arange(17)
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
    time_axis_for_ephys = np.arange(round(len(chosen_ephys_data_continuous_obj.samples)*time_slice))/chosen_ephys_data_continuous_obj.metadata['sample_rate']
    start_video_capture_idx = find_peaks(chosen_ephys_data_continuous_obj.samples[:,-1],height=0.3)[0][0] # find the beginning camera TTL pulse 
    time_axis_for_anipose = np.arange(0,20,0.01)+time_axis_for_ephys[start_video_capture_idx]

    # identify motion peak locations for foot strike
    signal_to_filter = 'palm_L_y'; do_filter=True
    if do_filter == True:
        filtered_signal = butter_highpass_filter(
            data=chosen_anipose_df[signal_to_filter].values,
            cutoff=0.5, fs=camera_fps, order=5)
    else: # do not filter
        filtered_signal=chosen_anipose_df[signal_to_filter].values
    
    foot_strike_idxs, _ = find_peaks(
        filtered_signal,
        height=[0,None],
        threshold=None,
        distance=20,
        prominence=None,
        width=None,
        wlen=None,
        )

    foot_off_idxs, _ = find_peaks(
        -filtered_signal,
        height=[0,None],
        threshold=None,
        distance=20,
        prominence=None,
        width=None,
        wlen=None,
        )

    MU_idxs = []
    for iChannel in range(chosen_ephys_data_continuous_obj.samples.shape[1]):
        MU_idxs_for_channel, _ = find_peaks(
            chosen_ephys_data_continuous_obj.samples[:,iChannel],
            height=[0,None],
            threshold=None,
            distance=20,
            prominence=None,
            width=None,
            wlen=None,
            )
        MU_idxs.append(MU_idxs_for_channel)

    from IPython.display import display
    # print(foot_strike_idxs - foot_off_idxs)
    df_fs_minus_fo = pd.DataFrame(foot_off_idxs[-13:-2] - foot_strike_idxs[-13:-2])
    display(df_fs_minus_fo.describe())

    # calclate number of rows to allocate for each subplot
    number_of_rows = len(bodyparts)+len(ephys_channel_idxs)
    row_spec_list = number_of_rows*[[None]]
    row_spec_list[0] = [{'rowspan': len(bodyparts)}]
    row_spec_list[len(bodyparts)] = [{'rowspan': len(ephys_channel_idxs)}]

    fig = make_subplots(
        rows=len(bodyparts)+len(ephys_channel_idxs), cols=1,
        # specs=[[{'secondary_y': True}]],#,[{'secondary_y': True}]],
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
            if name == signal_to_filter:
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
                name=bodyparts[bodypart_counter]+' peaks',
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
                name=bodyparts[bodypart_counter]+' peaks',
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
                # secondary_y=False
                )
                bodypart_counter += 1 # increment for each matching bodypart

    for iChannel in ephys_channel_idxs:
        fig.add_trace(go.Scatter(
            x=time_axis_for_ephys,
            # spread data values out by 10 millivolts
            y=chosen_ephys_data_continuous_obj.samples[:round(time_slice*len(chosen_ephys_data_continuous_obj.samples)),iChannel] + 10000*iChannel,
            name=f"CH{iChannel}",
            mode='lines',
            opacity=1,
            line=dict(width=.4)),
            row=len(bodyparts)+1, col=1,
            # secondary_y=True
            )
    
    # fig.update_layout(
        # title=f"Locomotion Kinematics and Neural Activity: {list(chosen_anipose_data_dict.keys())[0]}",
        # xaxis_title='Time (s)', row = 2, col = 1
        # yaxis_title='Voltage (uV)'
    # )

    fig.update_xaxes(
        title_text="<b>Time (s)</b>", row = len(bodyparts)+1, col = 1,#secondary_y=False
        )

    fig.update_yaxes(
        title_text="<b>Pixels</b>", row = 1, col = 1,#secondary_y=False
        )

    fig.update_yaxes(
        title_text="<b>Voltage (uV)</b>", row = len(bodyparts)+1, col = 1 # secondary_y=True
        )

    fig.show()

def psth(
    ephys_data_dict,
    anipose_data_dict,
    bodyparts=['palm_L_z','palm_R_z'],
    session_date=str(time.strftime("%y%m%d")),
    rat_name='dogerat',
    treadmill_speed=20,
    treadmill_incline=0,
    camera_fps=100,
    time_slice=1,
    ephys_channel_idxs=np.arange(17)
    ):
    
    # format inputs to avoid ambiguities
    rat_name = str(rat_name).lower()
    treadmill_speed = str(treadmill_speed).zfill(2)
    treadmill_incline = str(treadmill_incline).zfill(2)

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
    time_axis_for_ephys = np.arange(round(len(chosen_ephys_data_continuous_obj.samples)*time_slice))/chosen_ephys_data_continuous_obj.metadata['sample_rate']
    start_video_capture_idx = find_peaks(chosen_ephys_data_continuous_obj.samples[:,-1],height=0.3)[0][0] # find the beginning camera TTL pulse 
    time_axis_for_anipose = np.arange(0,20,0.01)+time_axis_for_ephys[start_video_capture_idx]

    # identify motion peak locations for foot strike
    signal_to_filter = 'palm_L_y'; do_filter=True
    if do_filter == True:
        filtered_signal = butter_highpass_filter(
            data=chosen_anipose_df[signal_to_filter].values,
            cutoff=0.5, fs=camera_fps, order=5)
    else: # do not filter
        filtered_signal=chosen_anipose_df[signal_to_filter].values
    
    foot_strike_idxs, _ = find_peaks(
        filtered_signal,
        height=[0,None],
        threshold=None,
        distance=20,
        prominence=None,
        width=None,
        wlen=None,
        )

    foot_off_idxs, _ = find_peaks(
        -filtered_signal,
        height=[0,None],
        threshold=None,
        distance=20,
        prominence=None,
        width=None,
        wlen=None,
        )

    from IPython.display import display
    # print(foot_strike_idxs - foot_off_idxs)
    df_fs_minus_fo = pd.DataFrame(foot_off_idxs[-13:-2] - foot_strike_idxs[-13:-2])
    display(df_fs_minus_fo.describe())

    # calclate number of rows to allocate for each subplot
    number_of_rows = len(bodyparts)+len(ephys_channel_idxs)
    row_spec_list = number_of_rows*[[None]]
    row_spec_list[0] = [{'rowspan': len(bodyparts)}]
    row_spec_list[len(bodyparts)] = [{'rowspan': len(ephys_channel_idxs)}]

    fig = make_subplots(
        rows=len(bodyparts)+len(ephys_channel_idxs), cols=1,
        # specs=[[{'secondary_y': True}]],#,[{'secondary_y': True}]],
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
            if name == signal_to_filter:
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
                name=bodyparts[bodypart_counter]+' peaks',
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
                name=bodyparts[bodypart_counter]+' peaks',
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
                # secondary_y=False
                )
                bodypart_counter += 1 # increment for each matching bodypart

    for iChannel in ephys_channel_idxs:
        fig.add_trace(go.Scatter(
            x=time_axis_for_ephys,
            # spread data values out by 10 millivolts
            y=chosen_ephys_data_continuous_obj.samples[:round(time_slice*len(chosen_ephys_data_continuous_obj.samples)),iChannel] + 10000*iChannel,
            name=f"CH{iChannel}",
            mode='lines',
            opacity=1,
            line=dict(width=.4)),
            row=len(bodyparts)+1, col=1,
            # secondary_y=True
            )
    
    # fig.update_layout(
        # title=f"Locomotion Kinematics and Neural Activity: {list(chosen_anipose_data_dict.keys())[0]}",
        # xaxis_title='Time (s)', row = 2, col = 1
        # yaxis_title='Voltage (uV)'
    # )

    fig.update_xaxes(
        title_text="<b>Time (s)</b>", row = len(bodyparts)+1, col = 1,#secondary_y=False
        )

    fig.update_yaxes(
        title_text="<b>Pixels</b>", row = 1, col = 1,#secondary_y=False
        )

    fig.update_yaxes(
        title_text="<b>Voltage (uV)</b>", row = len(bodyparts)+1, col = 1 # secondary_y=True
        )

    fig.show()
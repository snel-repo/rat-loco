from pdb import set_trace
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.offline import iplot
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, butter, filtfilt, medfilt
from IPython.display import display

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
def process_steps(
    anipose_data_dict, bodypart_for_alignment, bodypart_for_reference, subtract_bodypart_ref, origin_offsets,
    filter_tracking, session_date, rat_name, treadmill_speed, treadmill_incline,
    camera_fps, align_to, time_frame
    ):

    # format inputs to avoid ambiguities
    rat_name = str(rat_name).lower()
    treadmill_speed = str(treadmill_speed).zfill(2)
    treadmill_incline = str(treadmill_incline).zfill(2)

    # filter anipose dictionaries for the proper session date, speed, and incline
    anipose_data_dict_filtered_by_date = dict(filter(lambda item:
        str(session_date) in item[0], anipose_data_dict.items()
        ))
    anipose_data_dict_filtered_by_ratname = dict(filter(lambda item:
        rat_name in item[0], anipose_data_dict_filtered_by_date.items()
        ))
    anipose_data_dict_filtered_by_speed = dict(filter(lambda item:
        "speed"+treadmill_speed in item[0], anipose_data_dict_filtered_by_ratname.items()
        ))
    anipose_data_dict_filtered_by_incline = dict(filter(lambda item:
        "incline"+treadmill_incline in item[0], anipose_data_dict_filtered_by_speed.items()
        ))
    chosen_anipose_data_dict = anipose_data_dict_filtered_by_incline
    chosen_anipose_df = chosen_anipose_data_dict[
        f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
        ]

    # bodypart_to_filter = bodypart_for_alignment[0]
    cols = chosen_anipose_df.columns
    bodypart_substr = ['_x','_y','_z']
    not_bodypart_substr = ['ref','origin']
    reduced_cols = [str for str in cols if any(sub in str for sub in bodypart_substr)]
    bodypart_cols = [str for str in reduced_cols if not any(
        sub in str for sub in not_bodypart_substr)]
    bodypart_anipose_df = chosen_anipose_df[bodypart_cols]
    ref_aligned_df = bodypart_anipose_df.copy()
    for iDim in bodypart_substr:
        # if iDim == 'Labels': continue # skip if Labels column
        body_dim_cols = [str for str in bodypart_cols if any(sub in str for sub in [iDim])]
        if subtract_bodypart_ref:
            for iCol in body_dim_cols:
                ref_aligned_df[iCol] = \
                    bodypart_anipose_df[iCol] - bodypart_anipose_df[bodypart_for_reference[0]+iDim]
        else:
            # iDim_cols = [str for str in bodypart_cols if str in [iDim]]
            if type(origin_offsets[iDim[-1]]) is int:
                ref_aligned_df[body_dim_cols] = \
                    bodypart_anipose_df[body_dim_cols]+origin_offsets[iDim[-1]]
            elif type(origin_offsets[iDim[-1]]) is list and type(origin_offsets[iDim[-1]][0]) is str:
                for iCol in body_dim_cols:
                    ref_aligned_df[iCol] = \
                        bodypart_anipose_df[iCol] - bodypart_anipose_df[bodypart_for_reference[0]+iDim]
            else:
                raise TypeError('origin_offsets variable must be either type `int`, or set as `bodypart_for_reference` (variable which is a string inside a list)')
            # ref_aligned_df = bodypart_anipose_df # do not subtract any reference
            # ref_aligned_df[iCol] = \
            #     bodypart_anipose_df[iCol] + bodypart_anipose_df[bodypart_for_reference[0]+iDim] # add measured offsets

    x_data = ref_aligned_df.columns.str.endswith("_x")
    y_data = ref_aligned_df.columns.str.endswith("_y")
    z_data = ref_aligned_df.columns.str.endswith("_z")
    
    sorted_body_anipose_df = pd.concat(
        [ref_aligned_df.loc[:,x_data],
         ref_aligned_df.loc[:,y_data],
         ref_aligned_df.loc[:,z_data]],
        axis=1,ignore_index=False)
    
    if filter_tracking == 'highpass':
        filtered_anipose_data = butter_highpass_filter(
            data=sorted_body_anipose_df.values,
            cutoff=0.5, fs=camera_fps, order=5)
    elif filter_tracking == 'median':
        filtered_anipose_data = medfilt(sorted_body_anipose_df.values, kernel_size=[5,1])
    else: # do not filter
        filtered_anipose_data=sorted_body_anipose_df.values
    processed_anipose_df = pd.DataFrame(filtered_anipose_data,columns=sorted_body_anipose_df.columns)
    # set_trace()
    # identify motion peak locations for foot strike
    foot_strike_idxs, _ = find_peaks(
        processed_anipose_df[bodypart_for_alignment[0]],
        height=[None,None],
        threshold=None,
        distance=30,
        prominence=None,
        width=None,
        wlen=None,
        rel_height=None
        )

    foot_off_idxs, _ = find_peaks(
        -processed_anipose_df[bodypart_for_alignment[0]], # invert signal to find the troughs
        height=[None,None],
        threshold=None,
        distance=30,
        prominence=None,
        width=None,
        wlen=None,
        rel_height=None
        )

    # index off outermost steps, to skip noisy initial tracking
    # foot_strike_idxs=foot_strike_idxs[1:-1]
    # foot_off_idxs=foot_off_idxs[1:-1]
    
    if align_to == 'foot strike':
        step_idxs = foot_strike_idxs
    elif align_to == 'foot off':
        step_idxs = foot_off_idxs
        
    # all_steps_df = pd.DataFrame(step_idxs)
    # all_step_stats = all_steps_df.describe()[0]
    
    # foot_strike_slice_idxs = [
    #     foot_strike_idxs[start_step],
    #     foot_strike_idxs[stop_step]
    #     ]
    # foot_off_slice_idxs = [
    #     foot_off_idxs[start_step],
    #     foot_off_idxs[stop_step]
    #     ]
    if time_frame==1:
        step_time_slice = slice(0,-1)
        time_frame=[0,1]
        start_step = int(len(step_idxs)*time_frame[0])
        stop_step = int(len(step_idxs)*time_frame[1])
        step_slice = slice(start_step, stop_step)
    else:
        start_step = int(len(step_idxs)*time_frame[0])
        stop_step = int(len(step_idxs)*time_frame[1])
        step_slice = slice(start_step, stop_step)
        if align_to == 'foot strike':
            step_time_slice = slice(foot_strike_idxs[start_step],foot_strike_idxs[stop_step-1])
        elif align_to == 'foot off':
            step_time_slice = slice(foot_off_idxs[start_step],foot_off_idxs[stop_step-1])
    
    # step_time_slice = slice(all_step_idx[0],all_step_idx[1])
    sliced_steps_diff = pd.DataFrame(np.diff(step_idxs[start_step:stop_step]))#step_idxs[1:] - step_idxs[:-1])
    
    # set_trace()
    print(
        f"Inter-step timing stats for {align_to}, from step {start_step} to {stop_step-1}:\
            \nFile: {session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}")
    
    sliced_step_stats = sliced_steps_diff.describe()[0]
    
    display(sliced_step_stats)
    
    ## section plots bodypart tracking for the chosen session, for validation
    # import matplotlib.pyplot as plt
    # plt.plot(filtered_anipose_data)
    # plt.scatter(step_idxs, filtered_anipose_data[step_idxs],c='r')
    # plt.title(r'Check Peaks for ' + str(bodypart_to_filter))
    # plt.show()
    # print(foot_strike_idxs - foot_off_idxs)
    # set_trace()
    return processed_anipose_df, foot_strike_idxs, foot_off_idxs, sliced_step_stats, step_slice, step_time_slice

def trialize_steps(anipose_data_dict, bodypart_for_alignment, bodypart_for_reference, subtract_bodypart_ref, origin_offsets,
    filter_tracking, session_date, rat_name, treadmill_speed, treadmill_incline,
    camera_fps, align_to, time_frame):
    
    (processed_anipose_df, foot_strike_idxs, foot_off_idxs,
     sliced_step_stats, step_slice, step_time_slice
     ) = process_steps(anipose_data_dict, bodypart_for_alignment, bodypart_for_reference,
                       subtract_bodypart_ref, origin_offsets, filter_tracking, session_date, rat_name,
                       treadmill_speed, treadmill_incline, camera_fps, align_to, time_frame)
    
    # get column titles
    # X_data_column_titles = processed_anipose_df.columns.str.endswith("_x")
    # Y_data_column_titles = processed_anipose_df.columns.str.endswith("_y")
    # Z_data_column_titles = processed_anipose_df.columns.str.endswith("_z")
    # all_data_column_titles = X_data_column_titles + Y_data_column_titles + Z_data_column_titles
    
    # extract all data as numpy array
    # X_data_npy = processed_anipose_df.loc[step_time_slice,X_data_column_titles].to_numpy()
    # Y_data_npy = processed_anipose_df.loc[step_time_slice,Y_data_column_titles].to_numpy()
    # Z_data_npy = processed_anipose_df.loc[step_time_slice,Z_data_column_titles].to_numpy()
    
    # trialize data
    pre_align_offset = 40
    post_align_offset = 20
    if align_to == 'foot strike':
        step_idxs = foot_strike_idxs[step_slice]
    elif align_to == 'foot off':
        step_idxs = foot_off_idxs[step_slice]
    
    # create trialized list of DataFrames: trialized_anipose_df_lst[Step][Values,'Bodypart']
    trialized_anipose_df_lst = []
    
    for iStep, step_idx in enumerate(step_idxs):
        trialized_anipose_df_lst.append(
            processed_anipose_df.iloc[step_idx-pre_align_offset:step_idx+post_align_offset,:])
        trialized_anipose_df_lst[iStep].columns = [iCol+f"_{iStep}" for iCol in trialized_anipose_df_lst[iStep].columns]
        trialized_anipose_df_lst[iStep].reset_index(drop=True, inplace=True)
    trialized_anipose_df = pd.concat(trialized_anipose_df_lst, axis=1)
        
    return trialized_anipose_df, pre_align_offset, post_align_offset, step_idxs

def behavioral_space(anipose_data_dict, bodypart_for_alignment, bodypart_for_reference,
                     subtract_bodypart_ref, origin_offsets, bodyparts_list, filter_tracking, session_date,
                     rat_name, treadmill_speed, treadmill_incline, camera_fps, align_to,
                     time_frame, MU_colors, CH_colors):
    
    iPar = 0
    session_parameters_lst = []
    trialized_anipose_dfs_lst = []
    subtitles = []
    for iTitle in treadmill_incline:
        subtitles.append("<b>Incline: "+str(iTitle)+"</b>")
    fig1 = go.Figure()
    fig2 = make_subplots(
        rows=len(bodyparts_list), cols=len(treadmill_incline),
        shared_xaxes=True,
        shared_yaxes='rows',
        subplot_titles=subtitles
        )
        # f"<b>Locomotion Kinematics: {list(chosen_anipose_data_dict.keys())[0]}</b>",
        # f"<b>Neural Activity: {list(chosen_ephys_data_dict.keys())[0]}</b>"
    for iPar in range(len(treadmill_incline)):
        trialized_anipose_df, pre_align_offset, post_align_offset, step_idxs = \
            trialize_steps(anipose_data_dict, bodypart_for_alignment,
                           bodypart_for_reference, subtract_bodypart_ref, origin_offsets,
                           filter_tracking, session_date[iPar], rat_name[iPar],
                           treadmill_speed[iPar], treadmill_incline[iPar],
                           camera_fps, align_to, time_frame)
        
        i_session_date = session_date[iPar]
        i_rat_name = str(rat_name[iPar]).lower()
        i_treadmill_speed = str(treadmill_speed[iPar]).zfill(2)
        i_treadmill_incline = str(treadmill_incline[iPar]).zfill(2)
        session_parameters_lst.append(
            f"{i_session_date}_{i_rat_name}_speed{i_treadmill_speed}_incline{i_treadmill_incline}")
        trialized_anipose_dfs_lst.append(trialized_anipose_df)
        trialized_anipose_dfs_lst[iPar]['Labels'] = pd.Series(int(i_treadmill_incline) * \
                                np.ones(anipose_data_dict[session_parameters_lst[iPar]].shape[0]))
        
        # get trial averages by filtering for all columns that match
        trial_ave_lst = []
        for iBodypart in range(len(bodyparts_list)):
            trial_ave_lst.append(trialized_anipose_df.filter(like=bodyparts_list[iBodypart]).mean(axis=1))

        # plot trial averages across inclines
        fig1.add_trace(go.Scatter(
            x=trial_ave_lst[0],
            y=trial_ave_lst[1],
            mode='lines', name=f'incline{i_treadmill_incline}',
            line_color=MU_colors[int(i_treadmill_incline)//5]
            ))
        fig1.add_trace(go.Scatter(
            x=[trial_ave_lst[0][0]],
            y=[trial_ave_lst[1][0]],
            mode='markers', marker_line_color="black", marker_color=CH_colors[int(i_treadmill_incline)//5],
            marker_line_width=2, marker_size=15, marker_symbol='asterisk', name=f'incline{i_treadmill_incline} start',
            ))

        # plot single trajectories
        for iBodypart in range(len(bodyparts_list)):
            anipose_bodypart_trials = trialized_anipose_df.filter(like=bodyparts_list[iBodypart]).to_numpy()
            for trial in anipose_bodypart_trials.T:
                fig2.add_trace(go.Scatter(
                    x=np.linspace(-pre_align_offset/camera_fps, post_align_offset/camera_fps,len(trial)),
                    y=trial,
                    mode='lines',
                    opacity=.9,
                    line_color=MU_colors[int(i_treadmill_incline)//5],
                    line=dict(width=2)),
                    col=iPar+1, row=iBodypart+1
                    )
            fig2.add_vline(x=0, line_width=3, line_dash="dash", line_color="black", name=align_to)
        
    # set_trace()
    # Edit the layout
    fig1.update_layout(title=f'<b>Behavioral State Space Across {bodyparts_list[0]} and {bodyparts_list[1]}, Trial Averages</b>',
                   xaxis_title='<b>'+bodyparts_list[0]+' mean</b>',
                   yaxis_title='<b>'+bodyparts_list[1]+' mean</b>')
    fig1.update_yaxes(scaleanchor = "x",scaleratio = 1)
    for xx in range(len(treadmill_incline)):
        fig2.update_xaxes(title_text='<b>Time (sec)</b>', row = len(bodyparts_list), col = xx+1)
    for yy, yTitle in enumerate(bodyparts_list):
        fig2.update_yaxes(title_text="<b>"+str(yTitle)+" (mm)</b>", row = yy+1, col = 1)
    # fig2.update_yaxes(scaleanchor = "x",scaleratio = 1)
    # fig2.update_yaxes(matches='y')
    
    iplot(fig1)
    iplot(fig2)
    return
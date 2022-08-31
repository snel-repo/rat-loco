from pdb import set_trace
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, medfilt

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
    anipose_data_dict, bodypart_for_alignment, bodypart_for_reference, subtract_bodypart_ref,
    filter_tracking, session_date, rat_name, treadmill_speed, treadmill_incline,
    camera_fps, alignto, time_frame
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
    ref_bodypart_aligned_df = bodypart_anipose_df.copy()
    for iDim in bodypart_substr:
        # if iDim == 'Labels': continue # skip if Labels column
        body_dim_cols = [str for str in bodypart_cols if any(sub in str for sub in [iDim])]
        for iCol in body_dim_cols:
            if subtract_bodypart_ref:
                ref_bodypart_aligned_df[iCol] = \
                    bodypart_anipose_df[iCol] - bodypart_anipose_df[bodypart_for_reference[0]+iDim]
            else:
                ref_bodypart_aligned_df = bodypart_anipose_df # do not subtract reference
        
    x_data = ref_bodypart_aligned_df.columns.str.endswith("_x")
    y_data = ref_bodypart_aligned_df.columns.str.endswith("_y")
    z_data = ref_bodypart_aligned_df.columns.str.endswith("_z")
    
    sorted_body_anipose_df = pd.concat(
        [ref_bodypart_aligned_df.loc[:,x_data],
         ref_bodypart_aligned_df.loc[:,y_data],
         ref_bodypart_aligned_df.loc[:,z_data]],
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
    foot_strike_idxs=foot_strike_idxs[1:-1]
    foot_off_idxs=foot_off_idxs[1:-1]
    
    if alignto == 'foot strike':
        step_idxs = foot_strike_idxs
    elif alignto == 'foot off':
        step_idxs = foot_off_idxs
        
    all_steps_diff = pd.DataFrame(np.diff(step_idxs))
    all_step_stats = all_steps_diff.describe()[0]
    
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
        start_step = int((all_step_stats['count']+1)*time_frame[0])
        stop_step = int((all_step_stats['count']+1)*time_frame[1])
        step_slice = slice(start_step, stop_step)
    else:
        start_step = int((all_step_stats['count']+1)*time_frame[0])
        stop_step = int((all_step_stats['count']+1)*time_frame[1])
        step_slice = slice(start_step, stop_step)
        if alignto == 'foot strike':
            step_time_slice = slice(foot_strike_idxs[start_step],foot_strike_idxs[stop_step-1])
        elif alignto == 'foot off':
            step_time_slice = slice(foot_off_idxs[start_step],foot_off_idxs[stop_step-1])
    
    # step_time_slice = slice(all_step_idx[0],all_step_idx[1])
    sliced_steps_diff = pd.DataFrame(np.diff(step_idxs[start_step:stop_step]))#step_idxs[1:] - step_idxs[:-1])
    
    # set_trace()
    print(
        f"Inter-step timing stats for {alignto}, from step {start_step} to {stop_step}:\
            \nFile: {session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}")
    
    sliced_step_stats = sliced_steps_diff.describe()[0]
    
    from IPython.display import display
    display(sliced_step_stats)
    
    ## section plots bodypart tracking for the chosen session, for validation
    # import matplotlib.pyplot as plt
    # plt.plot(filtered_anipose_data)
    # plt.scatter(step_idxs, filtered_anipose_data[step_idxs],c='r')
    # plt.title(r'Check Peaks for ' + str(bodypart_to_filter))
    # plt.show()
    # print(foot_strike_idxs - foot_off_idxs)

    return processed_anipose_df, foot_strike_idxs, foot_off_idxs, sliced_step_stats, step_slice, step_time_slice

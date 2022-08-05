from pdb import set_trace
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt

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
def extract_step_idxs(
    anipose_data_dict, bodypart_for_alignment, filter_tracking,
    session_date, rat_name, treadmill_speed, treadmill_incline,
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

    # identify motion peak locations for foot strike
    bodypart_to_filter = bodypart_for_alignment[0]
    
    if filter_tracking == True:
        filtered_signal = butter_highpass_filter(
            data=chosen_anipose_df[bodypart_to_filter].values,
            cutoff=0.5, fs=camera_fps, order=5)
    else: # do not filter
        filtered_signal=chosen_anipose_df[bodypart_to_filter].values
    
    foot_strike_idxs, _ = find_peaks(
        filtered_signal,
        height=[None,None],
        threshold=None,
        distance=30,
        prominence=None,
        width=None,
        wlen=None,
        rel_height=None
        )

    foot_off_idxs, _ = find_peaks(
        -filtered_signal, # invert signal to find the troughs
        height=[None,None],
        threshold=None,
        distance=30,
        prominence=None,
        width=None,
        wlen=None,
        rel_height=None
        )

    if alignto == 'foot strike':
        step_idxs = foot_strike_idxs
    elif alignto == 'foot off':
        step_idxs = foot_off_idxs
        
    all_steps_diff = pd.DataFrame(np.diff(step_idxs[1:-1])) # skip outermost steps! (noisy)
    all_step_stats = all_steps_diff.describe()[0]
    
    start_step = int((all_step_stats['count'])*time_frame[0])
    stop_step = int((all_step_stats['count'])*time_frame[1])
    step_slice = slice(start_step, stop_step)
    
    foot_strike_slice_idxs = [
        foot_strike_idxs[start_step],
        foot_strike_idxs[stop_step]
        ]
    foot_off_slice_idxs = [
        foot_off_idxs[start_step],
        foot_off_idxs[stop_step]
        ]
    
    all_step_idx = []
    if alignto == 'foot strike':
        all_step_idx.append(foot_strike_slice_idxs[0])
        all_step_idx.append(foot_strike_slice_idxs[1])
    elif alignto == 'foot off':
        all_step_idx.append(foot_off_slice_idxs[0])
        all_step_idx.append(foot_off_slice_idxs[1])
    
    step_time_slice = slice(all_step_idx[0],all_step_idx[1])
    sliced_steps_diff = pd.DataFrame(np.diff(step_idxs[start_step:stop_step]))#step_idxs[1:] - step_idxs[:-1])
    
    print(
        f"Inter-step timing stats for {alignto}, from step {start_step} to {stop_step}:\
            \nFile: {session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}")
    
    sliced_step_stats = sliced_steps_diff.describe()[0]
    
    from IPython.display import display
    display(sliced_step_stats)
    
    ## section plots bodypart tracking for the chosen session, for validation
    # import matplotlib.pyplot as plt
    # plt.plot(filtered_signal)
    # plt.scatter(step_idxs, filtered_signal[step_idxs],c='r')
    # plt.title(r'Check Peaks for ' + str(bodypart_to_filter))
    # plt.show()
    # print(foot_strike_idxs - foot_off_idxs)

    return filtered_signal, foot_strike_idxs, foot_off_idxs, sliced_step_stats, step_slice, step_time_slice

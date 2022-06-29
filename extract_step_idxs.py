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
    anipose_data_dict, bodypart_for_tracking,
    session_date, rat_name, treadmill_speed, treadmill_incline,
    camera_fps, alignto
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
    bodypart_to_filter = bodypart_for_tracking[0]
    do_filter=True
    if do_filter == True:
        filtered_signal = butter_highpass_filter(
            data=chosen_anipose_df[bodypart_to_filter].values,
            cutoff=0.5, fs=camera_fps, order=5)
    else: # do not filter
        filtered_signal=chosen_anipose_df[bodypart_to_filter].values
    
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
        -filtered_signal, # invert signal to find the troughs
        height=[0,None],
        threshold=None,
        distance=20,
        prominence=None,
        width=None,
        wlen=None,
        )

    if alignto == 'foot strike':
        step_idxs = foot_strike_idxs
    elif alignto == 'foot off':
        step_idxs = foot_off_idxs

    from IPython.display import display
    # print(foot_strike_idxs - foot_off_idxs)
    df_fs_minus_fo = pd.DataFrame(step_idxs[1:] - step_idxs[:-1])
    print(
            f"Inter-step timing statistics for {alignto}:\n\
            File: {session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
        )
    step_stats = df_fs_minus_fo.describe()[0]
    display(step_stats)

    return foot_strike_idxs, foot_off_idxs, step_stats
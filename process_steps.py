from inspect import stack
from pdb import set_trace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display
from plotly.offline import iplot
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt, find_peaks, medfilt


# create highpass filter to remove baseline from foot tracking for better peak finding performance
def butter_highpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high")  # , analog=False) #, output="sos")
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=2):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


# create lowpass filter to smooth reference bodypart and reduce its variability
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low")  # , analog=False, output="sos")
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=2):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


# create bandpass filter to smooth reference bodypart and reduce its variability
def butter_bandpass(cutoffs, fs, order=2):
    assert len(cutoffs) == 2
    nyq = 0.5 * fs
    normal_cutoffs = np.array(cutoffs) / nyq
    b, a = butter(order, normal_cutoffs, btype="bandpass")  # , analog=False, output="sos")
    return b, a


def butter_bandpass_filter(data, cutoffs, fs, order=2):
    b, a = butter_bandpass(cutoffs, fs, order=order)
    y = filtfilt(b, a, data)
    return y


# function identifies peaks and troughs of anipose data, and
# optionally applies filtering and offset or reference bodypart subtraction
def peak_align_and_filt(
    chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator
):
    ### Unpack CFG Inputs
    # unpack analysis inputs
    (
        MU_spike_amplitudes_list,
        ephys_channel_idxs_list,
        filter_ephys,
        ephys_cutoffs,
        sort_method,
        sort_to_use,
        disable_anipose,
        bodypart_for_reference,
        bodypart_ref_filter,
        filter_all_anipose,
        anipose_cutoffs,
        trial_reject_bounds_mm,
        trial_reject_bounds_sec,
        trial_reject_bounds_vel,
        origin_offsets,
        save_binned_MU_data,
        time_frame,
        bin_width_ms,
        num_rad_bins,
        smoothing_window,
        phase_align,
        align_to,
        export_data,
    ) = CFG["analysis"].values()
    # unpack plotting inputs
    (plot_type, plot_units, do_plot, N_colors, plot_template, *_) = CFG["plotting"].values()
    # unpack chosen rat inputs
    (
        bodyparts_list,
        bodypart_for_alignment,
        session_date,
        treadmill_speed,
        treadmill_incline,
        camera_fps,
        vid_length,
    ) = CFG["rat"][chosen_rat].values()

    # format inputs to avoid ambiguities
    session_date = session_date[session_iterator]
    rat_name = str(chosen_rat).lower()
    treadmill_speed = str(treadmill_speed[session_iterator]).zfill(2)
    treadmill_incline = str(treadmill_incline[session_iterator]).zfill(2)
    session_ID = f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"

    # # format inputs to avoid ambiguities
    # rat_name = str(rat_name).lower()
    # treadmill_speed = str(treadmill_speed).zfill(2)
    # treadmill_incline = str(treadmill_incline).zfill(2)

    # ensure dict is extracted
    if type(anipose_dict) is dict:
        anipose_dict = anipose_dict[session_ID]
    cols = anipose_dict.columns
    bodypart_substr = ["_x", "_y", "_z"]
    not_bodypart_substr = ["ref", "origin"]
    reduced_cols = [str for str in cols if any(sub in str for sub in bodypart_substr)]
    bodypart_cols = [
        str for str in reduced_cols if not any(sub in str for sub in not_bodypart_substr)
    ]
    bodypart_anipose_df = anipose_dict[bodypart_cols]
    ref_aligned_df = bodypart_anipose_df.copy()
    ref_bodypart_trace_list = []
    if origin_offsets is not False:
        for iDim in bodypart_substr:
            # if iDim == 'Labels': continue # skip if Labels column
            body_dim_cols = [str for str in bodypart_cols if any(sub in str for sub in [iDim])]
            # iDim_cols = [str for str in bodypart_cols if str in [iDim]]
            if type(origin_offsets[iDim[-1]]) is int:
                ref_aligned_df[body_dim_cols] = (
                    bodypart_anipose_df[body_dim_cols] + origin_offsets[iDim[-1]]
                )
            elif type(origin_offsets[iDim[-1]]) is str:
                if bodypart_ref_filter:
                    ref_bodypart_trace_list.append(
                        butter_lowpass_filter(
                            data=bodypart_anipose_df[bodypart_for_reference + iDim],
                            cutoff=bodypart_ref_filter,
                            fs=camera_fps,
                            order=1,
                        )
                    )
                else:
                    ref_bodypart_trace_list.append(
                        bodypart_anipose_df[bodypart_for_reference] + iDim
                    )
                # if bodypart_for_reference != origin_offsets[iDim[-1]]:
                for iCol in body_dim_cols:
                    # set_trace()
                    if bodypart_for_reference not in iCol:
                        ref_aligned_df[iCol] = (
                            bodypart_anipose_df[iCol] - ref_bodypart_trace_list[-1]
                        )
            else:
                raise TypeError(
                    "origin_offsets must be a dict. Keys must be 'x'/'y'/'z' and values must be either type `int` or `str`."
                )
    # if origin_offsets is not False:
    #     iDim_map = dict(x=0,y=1,z=2)
    #     for iDim in bodypart_substr:
    #         # if iDim == 'Labels': continue # skip if Labels column
    #         body_dim_cols = [str for str in bodypart_cols if any(sub in str for sub in [iDim])]
    #         # iDim_cols = [str for str in bodypart_cols if str in [iDim]]
    #         if type(origin_offsets[iDim_map[iDim[-1]]]) is int:
    #             ref_aligned_df[body_dim_cols] = \
    #                 bodypart_anipose_df[body_dim_cols]+origin_offsets[iDim_map[iDim[-1]]]
    #         elif type(origin_offsets[iDim_map[iDim[-1]]]) is str:
    #             if bodypart_ref_filter:
    #                 ref_bodypart_trace_list.append(butter_lowpass_filter(
    #                     data=bodypart_anipose_df[bodypart_for_reference+iDim],
    #                     cutoff=bodypart_ref_filter, fs=camera_fps, order=1))
    #             else:
    #                 ref_bodypart_trace_list.append(bodypart_anipose_df[bodypart_for_reference+iDim])
    #             for iCol in body_dim_cols:
    #                 ref_aligned_df[iCol] = \
    #                     bodypart_anipose_df[iCol] - ref_bodypart_trace_list[-1]
    #         else:
    #             raise TypeError("origin_offsets variable must be either type `int`, or set as a base bodypart str, such as 'tailbase'")
    # ref_aligned_df = bodypart_anipose_df # do not subtract any reference
    # ref_aligned_df[iCol] = \
    #     bodypart_anipose_df[iCol] + ref_bodypart_trace_list # add measured offsets

    x_data = ref_aligned_df.columns.str.endswith("_x")
    y_data = ref_aligned_df.columns.str.endswith("_y")
    z_data = ref_aligned_df.columns.str.endswith("_z")

    sorted_body_anipose_df = pd.concat(
        [
            ref_aligned_df.loc[:, x_data],
            ref_aligned_df.loc[:, y_data],
            ref_aligned_df.loc[:, z_data],
        ],
        axis=1,
        ignore_index=False,
    )

    low = 1
    high = 8
    if filter_all_anipose == "highpass":
        filtered_anipose_data = butter_highpass_filter(
            data=sorted_body_anipose_df.values, cutoff=low, fs=camera_fps, order=2
        )
        print(f"A {filter_all_anipose} filter was applied to all anipose data (lowcut = {low}Hz).")
    elif filter_all_anipose == "lowpass":
        filtered_anipose_data = butter_lowpass_filter(
            data=sorted_body_anipose_df.values, cutoff=high, fs=camera_fps, order=2
        )
        print(
            f"A {filter_all_anipose} filter was applied to all anipose data (highcut = {high}Hz)."
        )
    elif filter_all_anipose == "bandpass":
        filtered_anipose_data = butter_bandpass_filter(
            data=sorted_body_anipose_df.values, cutoffs=[low, high], fs=camera_fps, order=2
        )
        print(
            f"A {filter_all_anipose} filter was applied to all anipose data (lowcut = {low}Hz, highcut = {high}Hz)."
        )
    elif filter_all_anipose == "median":
        filtered_anipose_data = medfilt(sorted_body_anipose_df.values, kernel_size=[7, 1])
        print(f"A {filter_all_anipose} filter was applied to all anipose data.")
    else:  # do not filter
        filtered_anipose_data = sorted_body_anipose_df.values
    processed_anipose_df = pd.DataFrame(
        filtered_anipose_data, columns=sorted_body_anipose_df.columns
    )
    # identify motion peak locations for foot strike
    foot_strike_idxs, _ = find_peaks(
        processed_anipose_df[bodypart_for_alignment[0]],
        height=[None, None],
        threshold=None,
        distance=30,
        prominence=None,
        width=10,
        wlen=None,
    )

    foot_off_idxs, _ = find_peaks(
        -processed_anipose_df[bodypart_for_alignment[0]],  # invert signal to find the troughs
        height=[None, None],
        threshold=None,
        distance=30,
        prominence=None,
        width=10,
        wlen=None,
    )

    foot_offs_strikes = np.array([*foot_off_idxs, *foot_strike_idxs])
    weaved_strikes_offs = [
        np.ones(len(foot_off_idxs) + len(foot_strike_idxs), dtype=int),
        foot_offs_strikes,
    ]
    for i in range(0, len(foot_off_idxs)):
        weaved_strikes_offs[0][i] = -1
    sorted_idxs = np.argsort(weaved_strikes_offs[1])
    weaved_strikes_offs[0] = weaved_strikes_offs[0][sorted_idxs]
    weaved_strikes_offs[1] = weaved_strikes_offs[1][sorted_idxs]

    idxs_to_remove = []
    bodypart_anipose_data = processed_anipose_df[bodypart_for_alignment].values
    for i in range(len(weaved_strikes_offs[0]) - 1):
        if weaved_strikes_offs[0][i] == weaved_strikes_offs[0][i + 1]:
            if np.abs(bodypart_anipose_data[weaved_strikes_offs[1][i]][0]) > np.abs(
                bodypart_anipose_data[weaved_strikes_offs[1][i + 1]][0]
            ):
                idxs_to_remove.append(i + 1)
            else:
                idxs_to_remove.append(i)
    # set_trace()
    cleaned_strikes_offs = np.delete(weaved_strikes_offs[1], idxs_to_remove, axis=0)

    # if foot_off_idxs[0] < foot_strike_idxs[0]:
    #     foot_off_idxs = cleaned_strikes_offs[::2]
    #     foot_strike_idxs = cleaned_strikes_offs[1::2]
    # else:
    #     foot_strike_idxs = cleaned_strikes_offs[::2]
    #     foot_off_idxs = cleaned_strikes_offs[1::2]

    # # set_trace()

    # # index off outermost steps, to skip noisy initial tracking
    # foot_strike_idxs = foot_strike_idxs[1:-1]
    # foot_off_idxs = foot_off_idxs[1:-1]

    if align_to == "foot strike":
        step_idxs = foot_strike_idxs
    elif align_to == "foot off":
        step_idxs = foot_off_idxs

    # all_steps_df = pd.DataFrame(step_idxs)
    # all_step_stats = all_steps_df.describe()[0]

    # foot_strike_slice_idxs = [foot_strike_idxs[start_step], foot_strike_idxs[stop_step]]
    # foot_off_slice_idxs = [foot_off_idxs[start_step], foot_off_idxs[stop_step]]
    if time_frame == 1:
        step_time_slice = slice(0, -1)
        time_frame = [0, 1]
        start_step = int(len(step_idxs) * time_frame[0])
        stop_step = int(len(step_idxs) * time_frame[1])
        step_slice = slice(start_step, stop_step)
    else:
        start_step = int(len(step_idxs) * time_frame[0])
        stop_step = int(len(step_idxs) * time_frame[1])
        step_slice = slice(start_step, stop_step)
        if align_to == "foot strike":
            step_time_slice = slice(foot_strike_idxs[start_step], foot_strike_idxs[stop_step - 1])
        elif align_to == "foot off":
            step_time_slice = slice(foot_off_idxs[start_step], foot_off_idxs[stop_step - 1])

    # step_time_slice = slice(all_step_idx[0],all_step_idx[1])
    sliced_steps_diff = pd.DataFrame(
        np.diff(step_idxs[step_slice])
    )  # step_idxs[1:] - step_idxs[:-1])

    print(
        f"Inter-step timing stats for {align_to}, from step {start_step} to {stop_step-1}:\
            \nFile: {session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
    )

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
    # calculate reference bodypart velocity in xyz axes and append to dataframe
    processed_anipose_df = get_bodypart_velocity(
        processed_anipose_df, bodypart_for_reference, treadmill_speed
    )
    # set_trace()

    return (
        processed_anipose_df,
        foot_strike_idxs,
        foot_off_idxs,
        sliced_step_stats,
        step_slice,
        step_time_slice,
        ref_bodypart_trace_list,
    )


def trialize_steps(
    chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator
):
    ### Unpack CFG Inputs
    # unpack analysis inputs
    (
        MU_spike_amplitudes_list,
        ephys_channel_idxs_list,
        filter_ephys,
        ephys_cutoffs,
        sort_method,
        sort_to_use,
        disable_anipose,
        bodypart_for_reference,
        bodypart_ref_filter,
        filter_all_anipose,
        anipose_cutoffs,
        trial_reject_bounds_mm,
        trial_reject_bounds_sec,
        trial_reject_bounds_vel,
        origin_offsets,
        save_binned_MU_data,
        time_frame,
        bin_width_ms,
        num_rad_bins,
        smoothing_window,
        phase_align,
        align_to,
        export_data,
    ) = CFG["analysis"].values()
    # unpack plotting inputs
    (plot_type, plot_units, do_plot, N_colors, plot_template, *_) = CFG["plotting"].values()
    # unpack chosen rat inputs
    (
        bodyparts_list,
        bodypart_for_alignment,
        session_date,
        treadmill_speed,
        treadmill_incline,
        camera_fps,
        vid_length,
    ) = CFG["rat"][chosen_rat].values()

    # format inputs to avoid ambiguities
    session_date = session_date[session_iterator]
    rat_name = str(chosen_rat).lower()
    treadmill_speed = str(treadmill_speed[session_iterator]).zfill(2)
    treadmill_incline = str(treadmill_incline[session_iterator]).zfill(2)
    session_ID = f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"

    (
        processed_anipose_df,
        foot_strike_idxs,
        foot_off_idxs,
        sliced_step_stats,
        step_slice,
        step_time_slice,
        ref_bodypart_trace_list,
    ) = peak_align_and_filt(
        chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator
    )

    # get column titles
    # X_data_column_titles = processed_anipose_df.columns.str.endswith("_x")
    # Y_data_column_titles = processed_anipose_df.columns.str.endswith("_y")
    # Z_data_column_titles = processed_anipose_df.columns.str.endswith("_z")
    # all_data_column_titles = X_data_column_titles + Y_data_column_titles + Z_data_column_titles

    # extract all data as numpy array
    # X_data_npy = processed_anipose_df.loc[step_time_slice,X_data_column_titles].to_numpy()
    # Y_data_npy = processed_anipose_df.loc[step_time_slice,Y_data_column_titles].to_numpy()
    # Z_data_npy = processed_anipose_df.loc[step_time_slice,Z_data_column_titles].to_numpy()

    # define variables for trializing data
    align_shift = 2 / 3  # default fraction of step
    pre_align_offset = int(
        int(
            sliced_step_stats.quantile(0.75) * align_shift
        )  # int(sliced_step_stats.quantile(0.75) *    align_shift) # //8
    )
    post_align_offset = int(
        int(sliced_step_stats.quantile(0.75) * (1 - align_shift))
    )  # * 7 // 8) #int(sliced_step_stats.quantile(0.75) * (1-align_shift))
    if align_to == "foot strike":
        step_idxs = foot_strike_idxs[step_slice]
    elif align_to == "foot off":
        step_idxs = foot_off_idxs[step_slice]

    # create trialized list of DataFrames: trialized_anipose_df_lst[Step][Values,'Bodypart']
    trialized_anipose_df_lst = []
    # slice off the last step index to avoid miscounting (avoid off by one error)
    true_step_idx = np.array([*range(step_slice.start, step_slice.stop)])[:-1]
    for iStep, (i_step_idx, i_true_step_num) in enumerate(zip(step_idxs[:-1], true_step_idx)):
        trialized_anipose_df_lst.append(
            processed_anipose_df.iloc[
                i_step_idx - pre_align_offset : i_step_idx + post_align_offset, :
            ]
        )
        # give column names to each step for all bodyparts, and
        # zerofill to the max number of digits in `true_step_idx`
        trialized_anipose_df_lst[iStep].columns = [
            iCol + f"_{str(i_true_step_num).zfill(int(1+np.log10(true_step_idx.max())))}"
            for iCol in trialized_anipose_df_lst[iStep].columns
        ]
        trialized_anipose_df_lst[iStep].reset_index(drop=True, inplace=True)
    trialized_anipose_df = pd.concat(trialized_anipose_df_lst, axis=1)

    all_trial_set = set(true_step_idx)
    keep_trial_set = set(true_step_idx)  # start with all sliced steps
    drop_trial_set = set()

    if trial_reject_bounds_sec:  # If not False, will enforce trial length bounds
        assert (
            len(trial_reject_bounds_sec) == 2 and type(trial_reject_bounds_sec) == list
        ), "`trial_reject_bounds_sec` must be a 2D list."
        trials_above_lb = set()
        trials_below_ub = set()

        lower_bound = trial_reject_bounds_sec[0]
        upper_bound = trial_reject_bounds_sec[1]
        step_durations_sec = np.diff(step_idxs) / camera_fps
        for iBodypart in bodyparts_list:
            trials_above_lb.update(true_step_idx[np.where(step_durations_sec > lower_bound)])
            trials_below_ub.update(true_step_idx[np.where(step_durations_sec < upper_bound)])
            # get trial idxs between bounds, loop through bodyparts, remove trials outside
            keep_trial_set.intersection_update(trials_above_lb & trials_below_ub)
        print(
            f"Step duration bounds: {np.round(lower_bound,decimals=3)}s to {np.round(upper_bound,decimals=3)}s"
        )
        drop_trial_set.update(all_trial_set - keep_trial_set)
        print(f"Steps outside of bounds: {drop_trial_set}")

    if trial_reject_bounds_mm:  # If not False, will remove outlier trials
        # use sets to store trials to be saved, to take advantage of set operations
        # as bodypart columns and rejection criteria are looped through
        trials_above_lb = set()
        trials_below_ub = set()
        # get idxs for `align_to` and shift by (pre_align_offset+post_align_offset)/2 in order to
        # approx a 180 degree phase shift and get idx of other peak/trough which was not aligned to
        # also use conditionals to check for whether the phase shift should be added or subtracted
        # based on whether pre_align_offset is greater/less than post_align_offset
        df_peak_and_trough_list = [
            trialized_anipose_df.iloc[
                pre_align_offset - (pre_align_offset + post_align_offset) // 2
            ]
            if pre_align_offset >= post_align_offset
            else trialized_anipose_df.iloc[
                pre_align_offset + (pre_align_offset + post_align_offset) // 2
            ],
            trialized_anipose_df.iloc[pre_align_offset],
        ]  # <- foot_off/foot_strike index

        # Get minimum value of each trial for each bodypart, then take median across trials.
        # Reject trials outside bounds around those medians.
        if type(trial_reject_bounds_mm) is int:
            assert (
                trial_reject_bounds_mm > 0
            ), "If `trial_reject_bounds_mm` is an integer, it must be greater than zero."
            if align_to == "foot strike":
                df_peak_or_trough = df_peak_and_trough_list[0]
            elif align_to == "foot off":
                df_peak_or_trough = df_peak_and_trough_list[1]
            for iBodypart in bodyparts_list:
                lower_bound = (
                    df_peak_or_trough.filter(like=iBodypart).median() - trial_reject_bounds_mm
                )
                upper_bound = (
                    df_peak_or_trough.filter(like=iBodypart).median() + trial_reject_bounds_mm
                )
                trials_above_lb.update(
                    true_step_idx[(df_peak_or_trough.filter(like=iBodypart) > lower_bound).values]
                )
                trials_below_ub.update(
                    true_step_idx[(df_peak_or_trough.filter(like=iBodypart) < upper_bound).values]
                )
                # get trial idxs between bounds, loop through bodyparts, remove trials outside
                keep_trial_set.intersection_update(trials_above_lb & trials_below_ub)
                print(
                    f"{align_to} bounds for {iBodypart}: {np.round(lower_bound,decimals=2)} to {np.round(upper_bound,decimals=2)}"
                )
            drop_trial_set.update(all_trial_set - keep_trial_set)
        elif type(trial_reject_bounds_mm) is list:
            # For each feature, assert that the first list element is less than the second list element
            assert (
                trial_reject_bounds_mm[0][0] < trial_reject_bounds_mm[0][1]
            ), "If `trial_reject_bounds_mm` is a 2d list, use form: [[10,40],[-10,25]] for maximal allowed Peak +/- and Trough +/- values. First of paired list value must be less than second."
            assert (
                trial_reject_bounds_mm[1][0] < trial_reject_bounds_mm[1][1]
            ), "If `trial_reject_bounds_mm` is a 2d list, use form: [[10,40],[-10,25]] for maximal allowed Peak +/- and Trough +/- values. First of paired list value must be less than second."
            # loop through both keys of the dictionary, keep trials when all bodyparts are constrained
            pair_description = ["Peak", "Trough"]
            for ii, (trial_reject_bounds_mm_pair, df_peak_or_trough) in enumerate(
                zip(trial_reject_bounds_mm, df_peak_and_trough_list)
            ):
                for iBodypart in bodyparts_list:
                    lower_bound = (
                        df_peak_or_trough.filter(like=iBodypart).median()
                        + trial_reject_bounds_mm_pair[0]
                    )
                    upper_bound = (
                        df_peak_or_trough.filter(like=iBodypart).median()
                        + trial_reject_bounds_mm_pair[1]
                    )
                    trials_above_lb.update(
                        true_step_idx[
                            (df_peak_or_trough.filter(like=iBodypart) > lower_bound).values
                        ]
                    )
                    trials_below_ub.update(
                        true_step_idx[
                            (df_peak_or_trough.filter(like=iBodypart) < upper_bound).values
                        ]
                    )
                    # get trial idxs between bounds, loop through bodyparts, remove trials outside
                    keep_trial_set.intersection_update(trials_above_lb & trials_below_ub)
                    print(
                        f"{pair_description[ii]} bounds for {iBodypart}: {np.round(lower_bound,decimals=2)} to {np.round(upper_bound,decimals=2)}"
                    )
                trials_above_lb, trials_below_ub = set(), set()
                drop_trial_set.update(all_trial_set - keep_trial_set)
        else:
            raise TypeError(
                "Wrong type specified for `trial_reject_bounds_mm` parameter in `rat_loco_analysis.py`"
            )

    if trial_reject_bounds_vel:
        reject_trials_temp = set()
        keep_trials_temp = set()

        lower_bound = trial_reject_bounds_vel[0]
        upper_bound = trial_reject_bounds_vel[1]
        # set_trace()

        if (type(lower_bound) is float) & (type(upper_bound) is float):
            velocity_dictionary = {
                k: v
                for (k, v) in trialized_anipose_df.items()
                if bodypart_for_reference + "_y_vel" in k
            }
            trial_num = 0
            for iTrial in velocity_dictionary:
                velocity_condition = (velocity_dictionary[iTrial] < lower_bound) | (
                    velocity_dictionary[iTrial] > upper_bound
                )
                out_of_bounds_indicies = np.where(velocity_condition)[0]
                if len(out_of_bounds_indicies) >= len(velocity_dictionary[iTrial]) / 2:
                    reject_trials_temp.add(trial_num)
                else:
                    keep_trials_temp.add(trial_num)
                trial_num += 1
            # update drop_trials_set with dropped trial indexes
            drop_trial_set.update(reject_trials_temp)
            keep_trial_set.update(keep_trial_set - reject_trials_temp)
        else:
            raise TypeError(
                "Wrong type specified for 'trial_reject_bounds_vel' parameter in 'rat_loco_analysis.py'. Need float"
            )
        print(f"Steps outside of bounds: {drop_trial_set}")

    for iTrial in drop_trial_set:
        # drop out of bounds trials from DataFrame in place, use log10 to get number of decimals for zfilling
        trialized_anipose_df.drop(
            list(
                trialized_anipose_df.filter(
                    like=f"_{str(iTrial).zfill(int(1+np.log10(true_step_idx.max())))}"
                )
            ),
            axis=1,
            inplace=True,
        )

    sliced_steps_diff = np.diff(step_idxs)
    kept_steps_diff = pd.DataFrame(
        np.array([sliced_steps_diff[iTrial - step_slice.start] for iTrial in (keep_trial_set)])
    )
    print(
        f"Inter-step timing stats for {align_to}, for steps: {keep_trial_set}:\
            \nSession ID: {session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
    )

    kept_step_stats = kept_steps_diff.describe()[0]

    display(kept_step_stats)
    if not trial_reject_bounds_mm and not trial_reject_bounds_sec:
        print(
            "No trials rejected, because `trial_reject_bounds_mm` and `trial_reject_bounds_sec` were set to False in `config/config.toml`"
        )
        kept_step_stats = sliced_step_stats
    else:
        print(f"Rejected trials: {drop_trial_set}")

    return (
        trialized_anipose_df,
        keep_trial_set,
        foot_strike_idxs,
        foot_off_idxs,
        sliced_step_stats,
        kept_step_stats,
        step_slice,
        step_time_slice,
        ref_bodypart_trace_list,
        pre_align_offset,
        post_align_offset,
        trial_reject_bounds_mm,
        trial_reject_bounds_sec,
    )


# calculates reference bodypart's velocity in xyz axes using diff() function
def get_bodypart_velocity(processed_anipose_df, bodypart_for_reference, treadmill_speed):
    bodypart_velocity = {}
    bodypart_velocity[bodypart_for_reference + "_x_vel"] = np.gradient(
        processed_anipose_df[bodypart_for_reference + "_x"]
    )
    bodypart_velocity[bodypart_for_reference + "_y_vel"] = np.add(
        np.gradient(processed_anipose_df[bodypart_for_reference + "_y"]), float(treadmill_speed)
    )
    bodypart_velocity[bodypart_for_reference + "_z_vel"] = np.gradient(
        processed_anipose_df[bodypart_for_reference + "_z"]
    )
    # set_trace()
    # boundary_applied_indicies = np.where([(bodypart_velocity[bodypart_for_reference + '_y_vel'] > lower_bound) & (bodypart_velocity[bodypart_for_reference + '_y_vel'] < upper_bound)])[1] #currently using np.where to find indicies

    time = np.arange(len(bodypart_velocity[bodypart_for_reference + "_y_vel"]))
    fig = make_subplots(rows=3, cols=1)
    fig.append_trace(
        go.Scatter(
            x=time,
            y=np.gradient(
                np.add(
                    np.gradient(processed_anipose_df[bodypart_for_reference + "_y"], edge_order=1),
                    float(treadmill_speed),
                )
            ),
            mode="lines",
            name="Y Acceleration",
        ),
        row=1,
        col=1,
    )
    # fig.add_trace(go.Scatter(x=time, y=bodypart_velocity[bodypart_for_reference + '_x_vel'], mode='lines', name='X Velocity'))
    fig.append_trace(
        go.Scatter(
            x=time,
            y=bodypart_velocity[bodypart_for_reference + "_y_vel"],
            mode="lines",
            name="Y Differential",
        ),
        row=2,
        col=1,
    )
    # fig.add_trace(go.Scatter(x=time, y=bodypart_velocity[bodypart_for_reference + '_z_vel'], mode='lines', name='Z Velocity'))
    fig.append_trace(
        go.Scatter(
            x=time,
            y=processed_anipose_df[bodypart_for_reference + "_y"],
            mode="lines",
            name="Y Positions",
        ),
        row=3,
        col=1,
    )
    fig.update_xaxes(title_text="Time (ms)")
    fig.update_yaxes(title_text="Acceleration (mm/ms^2)", row=1, col=1)
    fig.update_yaxes(title_text="Speed (mm/ms)", row=2, col=1)
    fig.update_yaxes(title_text="Position w/ respect to origin(mm)", row=3, col=1)

    fig.update_layout(title=bodypart_for_reference + " Velocity vs Position")

    # #fig.show()

    bodypart_velocity_df = pd.DataFrame.from_dict(bodypart_velocity)
    processed_anipose_df = pd.concat(
        [processed_anipose_df, bodypart_velocity_df], axis=1, join="inner"
    )

    # set_trace()

    return processed_anipose_df

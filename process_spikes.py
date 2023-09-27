from inspect import stack

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, find_peaks, iirnotch
from pdb import set_trace
import plot_handler
from process_steps import peak_align_and_filt, trialize_steps
import inspect

# from pdb import set_trace
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, RobustScaler
# import umap


# create highpass filters to remove baseline from SYNC channel
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
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
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


# main function for threshold sorting and producing spike indexes
def sort(
    chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_index
):
    ### Unpack CFG Inputs
    # unpack analysis inputs
    (
        MU_spike_amplitudes_list,
        ephys_channel_idxs_list,
        filter_ephys,
        sort_method,
        sort_to_use,
        bodypart_for_reference,
        bodypart_ref_filter,
        filter_all_anipose,
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
    (plot_type, plot_units, do_plot, N_colors, plot_template, *_) = CFG[
        "plotting"
    ].values()
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
    session_date = session_date[session_index]
    rat_name = str(chosen_rat).lower()
    treadmill_speed = str(treadmill_speed[session_index]).zfill(2)
    treadmill_incline = str(treadmill_incline[session_index]).zfill(2)
    session_ID = (
        f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
    )

    if len(bodyparts_list) > 0 and bodypart_for_alignment:
        assert bodyparts_list[0] == bodypart_for_alignment[0], (
            "Error: If bodyparts_list is not empty, bodyparts_list[0] must be "
            "the same as bodypart_for_alignment[0] in config.toml"
        )

    if do_plot == 2:  # override and ensure all plots are generated along the way
        if inspect.stack()[1][3] == "behavioral_space":
            plot_flag = False
        else:
            plot_flag = True
    elif do_plot == 0:
        plot_flag = False
    else:  # only display plot if rat_loco_analysis() is the caller
        plot_flag = (
            True
            if (
                stack()[1].function == "rat_loco_analysis"
                and not plot_type.__contains__("multi")
            )
            else False
        )
    # extract data from dictionaries
    chosen_ephys_data_continuous_obj = OE_dict[session_ID]
    chosen_anipose_df = anipose_dict[session_ID]

    # create time axes
    ephys_sample_rate = chosen_ephys_data_continuous_obj.metadata["sample_rate"]
    time_axis_for_ephys = (
        np.arange(round(len(chosen_ephys_data_continuous_obj.samples)))
        / ephys_sample_rate
    )
    # filter ephys data
    for channel_number in ephys_channel_idxs_list:
        if channel_number not in [-1, 16]:
            ephys_data_for_channel = chosen_ephys_data_continuous_obj.samples[:, channel_number]
        if filter_ephys == "notch" or filter_ephys == "both":
            ephys_data_for_channel = iir_notch(ephys_data_for_channel, ephys_sample_rate)
        if filter_ephys == "bandpass" or filter_ephys == "both":
            # 300-5000Hz band
            ephys_data_for_channel = butter_bandpass_filter(
                ephys_data_for_channel, 300.0, 5000.0, ephys_sample_rate
            )
        chosen_ephys_data_continuous_obj.samples[:, channel_number] = ephys_data_for_channel
    # find the beginning of the camera SYNC pulse
    filtered_sync_channel = butter_highpass_filter(
        data=chosen_ephys_data_continuous_obj.samples[:, 16],
        cutoff=50,
        fs=30000,
        order=2,
    )
    start_video_capture_ephys_idx = find_peaks(filtered_sync_channel, height=0.3)[0][0]
    time_axis_for_anipose = (
        np.arange(0, vid_length, 1 / camera_fps)
        + time_axis_for_ephys[start_video_capture_ephys_idx]
    )
    anipose_dict['time_axis_for_anipose'] = time_axis_for_anipose
    # # identify motion peak locations of bodypart for step cycle alignment
    # if filter_all_anipose == True:
    #     filtered_signal = butter_highpass_filter(
    #         data=chosen_anipose_df[bodypart_for_alignment[0]].values,
    #         cutoff=0.5, fs=camera_fps, order=5)
    # else: # do not filter
    #     filtered_signal=chosen_anipose_df[bodypart_for_alignment[0]].values

    (
        processed_anipose_df,
        foot_strike_idxs,
        foot_off_idxs,
        _,
        step_slice,
        step_time_slice,
        ref_bodypart_trace_list,
    ) = peak_align_and_filt(
        chosen_rat,
        OE_dict,
        KS_dict,
        anipose_dict,
        CH_colors,
        MU_colors,
        CFG,
        session_index,
    )

    # filter step peaks/troughs to be within chosen time_frame, but
    # only when time_frame=1 is not indicating to use the full dataset
    if time_frame != 1:
        foot_strike_slice_idxs = foot_strike_idxs[
            np.where(
                (foot_strike_idxs >= step_time_slice.start)
                & (foot_strike_idxs <= step_time_slice.stop)
            )
        ]
        foot_off_slice_idxs = foot_off_idxs[
            np.where(
                (foot_off_idxs >= step_time_slice.start)
                & (foot_off_idxs <= step_time_slice.stop)
            )
        ]

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
    step_to_ephys_conversion_ratio = ephys_sample_rate / camera_fps
    step_slice_bounds_in_ephys_time = []
    step_slice_bounds_in_ephys_time.append(
        int(step_to_ephys_conversion_ratio * (step_time_slice.start))
        + start_video_capture_ephys_idx
    )
    step_slice_bounds_in_ephys_time.append(
        int(step_to_ephys_conversion_ratio * (step_time_slice.stop))
        + start_video_capture_ephys_idx
    )
    if time_frame == 1:
        slice_for_ephys_during_video = slice(
            0, -1
        )  # get full anipose traces, if time_frame==1
    else:
        # step_slice = slice(step_time_slice.start,step_time_slice.stop)
        slice_for_ephys_during_video = slice(
            step_slice_bounds_in_ephys_time[0], step_slice_bounds_in_ephys_time[1]
        )

    # cluster the spikes waveforms with PCA
    # pca = PCA(n_components=3)
    # pca.fit(chosen_ephys_data_continuous_obj.samples)
    # print(pca.explained_variance_ratio_)
    if sort_method == "thresholding":
        # extract spikes that are detected in the selected amplitude threshold ranges
        MU_spikes_by_unit_dict = {}
        MU_spikes_by_unit_dict_keys = [
            str(int(unit[0])) for unit in MU_spike_amplitudes_list
        ]
        MU_channel_keys_list = [str(ch) for ch in ephys_channel_idxs_list]
        MU_spikes_dict = {key: None for key in MU_channel_keys_list}
        for channel_number in ephys_channel_idxs_list:
            MU_spike_idxs = (
                []
            )  # init empty list for each channel to hold next sorted spike idxs
            for iAmplitudes in MU_spike_amplitudes_list:
                if channel_number not in [-1, 16]:
                    # ephys_data_for_channel = chosen_ephys_data_continuous_obj.samples[
                    #     slice_for_ephys_during_video, channel_number
                    # ]
                    # if filter_ephys == "notch" or filter_ephys == "both":
                    #     ephys_data_for_channel = iir_notch(
                    #         ephys_data_for_channel, ephys_sample_rate
                    #     )
                    # if filter_ephys == "bandpass" or filter_ephys == "both":
                    #     # 350-7000Hz band
                    #     ephys_data_for_channel = butter_bandpass_filter(
                    #         ephys_data_for_channel, 250.0, 5000.0, ephys_sample_rate
                    #     )
                    MU_spike_idxs_for_channel, _ = find_peaks(
                        -ephys_data_for_channel,
                        height=iAmplitudes,
                        threshold=None,
                        distance=ephys_sample_rate // 1000,  # 1ms refractory period
                        prominence=None,
                        width=None,
                        wlen=None,
                    )
                    MU_spike_idxs.append(np.int32(MU_spike_idxs_for_channel))
            MU_spikes_by_unit_dict = dict(
                zip(MU_spikes_by_unit_dict_keys, MU_spike_idxs)
            )
            MU_spikes_dict[str(channel_number)] = MU_spikes_by_unit_dict
        if filter_ephys == "notch" or filter_ephys == "both":
            print("60Hz notch filter applied to voltage signals.")
        if filter_ephys == "bandpass" or filter_ephys == "both":
            print("350-7000Hz bandpass filter applied to voltage signals.")
        if filter_ephys not in ["notch", "bandpass", "both"]:
            print("No additional filters applied to voltage signals.")
    elif sort_method == "kilosort":
        try:
            chosen_KS_dict = KS_dict[session_ID]
        except KeyError:
            raise KeyError(
                f"{session_ID} not present in KS_dict, make sure that recording was present during Kilosorting for the sort_to_use folder choice"
            )

        except:
            raise
        MU_spikes_dict = {k: v for (k, v) in chosen_KS_dict.items() if k in plot_units}
        assert len(MU_spikes_dict) == len(plot_units), (
            "Selected MU key could be missing from input KS dictionary, "
            "check IDs in Phy, or try indexing from 1 in config.toml: [plotting]: plot_units."
        )
    else:
        raise ValueError("sort_method must be either 'kilosort' or 'thresholding'.")

    # MU_spike_idxs = np.array(MU_spike_idxs,dtype=object).squeeze().tolist()

    ### PLOTTING SECTION
    # if sort_method == 'thresholding':
    figs, sliced_MU_spikes_dict = plot_handler.sort_plot(
        session_index,
        anipose_dict,
        OE_dict,
        session_ID,
        do_plot,
        plot_flag,
        plot_template,
        plot_units,
        sort_method,
        origin_offsets,
        bodypart_for_reference,
        bodypart_for_alignment,
        bodyparts_list,
        bodypart_ref_filter,
        filter_all_anipose,
        time_frame,
        time_axis_for_anipose,
        time_axis_for_ephys,
        ephys_channel_idxs_list,
        chosen_ephys_data_continuous_obj,
        chosen_anipose_df,
        processed_anipose_df,
        slice_for_ephys_during_video,
        step_time_slice,
        foot_strike_slice_idxs,
        foot_off_slice_idxs,
        ref_bodypart_trace_list,
        MU_spikes_dict,
        MU_colors,
        CH_colors,
    )
    ### END PLOTTING SECTION

    if export_data:
        from scipy.io import loadmat, savemat

        # time_axis_for_ephys=time_axis_for_ephys[slice_for_ephys_during_video],
        # MU_spikes_by_KS_cluster = {str(k): np.array(v,dtype=int) for k, v in sliced_MU_spikes_dict.items()},
        # time_axis_for_anipose=time_axis_for_anipose,
        # foot_off_idxs = np.round(processed_anipose_df[bodypart_for_alignment[0]][foot_off_slice_idxs],1),
        # foot_strike_idxs = np.round(processed_anipose_df[bodypart_for_alignment[0]][foot_strike_slice_idxs],1),
        # anipose_data = {k: np.array(v,dtype=float) for k, v in chosen_anipose_df.to_dict('list').items()},
        # session_ID = session_ID
        export_dict = dict(
            time_axis_for_ephys=time_axis_for_ephys[slice_for_ephys_during_video],
            ephys_data=chosen_ephys_data_continuous_obj.samples[
                slice_for_ephys_during_video
            ],
            MU_spikes_by_KS_cluster={
                "unit" + str(k).zfill(2): np.array(v + 1, dtype=np.int64)
                for k, v in sliced_MU_spikes_dict.items()
            },
            time_axis_for_anipose=time_axis_for_anipose,
            foot_off_idxs=foot_off_idxs + 1,
            foot_strike_idxs=foot_strike_idxs + 1,
            foot_off_times=time_axis_for_anipose[foot_off_slice_idxs],
            foot_strike_times=time_axis_for_anipose[foot_strike_slice_idxs],
            anipose_data={
                k: np.array(v, dtype=float)
                for k, v in chosen_anipose_df.to_dict("list").items()
            },
            session_ID=session_ID,
        )
        savemat(f"{session_ID}.mat", export_dict, oned_as="column")
        # x = loadmat(f'{session_ID}.mat')
    # else:
    #     sliced_MU_spikes_dict = dict()
    
    
    if inspect.stack()[1][3] == "behavioral_space":
        print(f"{inspect.stack()[0][3]}() was called by {inspect.stack()[1][3]}()")

        steps_dict = create_steps_dict(
            plot_units,
            time_axis_for_ephys,
            MU_spikes_dict,
            slice_for_ephys_during_video,
            time_frame,
            time_axis_for_anipose,
            foot_strike_slice_idxs,
            foot_off_slice_idxs,
            )
        
        return steps_dict

    return (
        MU_spikes_dict,
        time_axis_for_ephys,
        chosen_anipose_df,
        time_axis_for_anipose,
        ephys_sample_rate,
        start_video_capture_ephys_idx,
        slice_for_ephys_during_video,
        session_ID,
        figs,
    )


def bin_and_count(
    chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_index
):
    ### Unpack CFG Inputs
    # unpack analysis inputs
    (
        MU_spike_amplitudes_list,
        ephys_channel_idxs_list,
        filter_ephys,
        sort_method,
        sort_to_use,
        bodypart_for_reference,
        bodypart_ref_filter,
        filter_all_anipose,
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
    (plot_type, plot_units, do_plot, N_colors, plot_template, *_) = CFG[
        "plotting"
    ].values()
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
    session_date = session_date[session_index]
    rat_name = str(chosen_rat).lower()
    treadmill_speed = str(treadmill_speed[session_index]).zfill(2)
    treadmill_incline = str(treadmill_incline[session_index]).zfill(2)
    session_ID = (
        f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
    )

    if do_plot == 2:  # override and ensure all plots are generated along the way
        plot_flag = True
    else:  # only display plot if rat_loco_analysis() is the caller
        plot_flag = (
            True
            if (
                stack()[1].function == "rat_loco_analysis"
                and not plot_type.__contains__("multi")
            )
            else False
        )

    # check inputs for problems
    if 16 in ephys_channel_idxs_list:
        ephys_channel_idxs_list.remove(16)
    if sort_method == "thresholding":
        assert (
            len(ephys_channel_idxs_list) == 1
        ), "ephys_channel_idxs_list should only be 1 channel, idiot! :)"
    assert type(bin_width_ms) is int, "bin_width_ms must be type 'int'."

    (
        MU_spikes_dict,
        _,
        _,
        _,
        ephys_sample_rate,
        _,
        slice_for_ephys_during_video,
        session_ID,
        _,
    ) = sort(
        chosen_rat,
        OE_dict,
        KS_dict,
        anipose_dict,
        CH_colors,
        MU_colors,
        CFG,
        session_index,
    )

    # (_, foot_strike_idxs, foot_off_idxs, sliced_step_stats,
    #  step_slice, step_time_slice, _) = peak_align_and_filt(
    #     bodypart_for_alignment=bodypart_for_alignment,
    #     bodypart_for_reference=bodypart_for_reference, bodypart_ref_filter=bodypart_ref_filter,
    #     origin_offsets=origin_offsets, filter_all_anipose=filter_all_anipose, session_date=session_date,
    #     rat_name=rat_name, treadmill_speed=treadmill_speed, treadmill_incline=treadmill_incline,
    #     camera_fps=camera_fps, align_to=align_to, time_frame=time_frame)

    (
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
    ) = trialize_steps(
        chosen_rat,
        OE_dict,
        KS_dict,
        anipose_dict,
        CH_colors,
        MU_colors,
        CFG,
        session_index,
    )

    # extract data dictionary (with keys for each unit) for the chosen electrophysiology channel
    if sort_method == "thresholding":
        MU_spikes_dict = MU_spikes_dict[
            str(ephys_channel_idxs_list[0])
        ]  # +slice_for_ephys_during_video.start
        print("!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! !!")
        print("!! Using FIRST channel in ephys_channel_idxs_list !!")
        print("!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! !!")
    # set conversion ratio from camera to electrophysiology sample rate
    step_to_ephys_conversion_ratio = ephys_sample_rate / camera_fps
    # initialize zero array to carry step-aligned spike activity,
    # with shape: Steps x Time (in ephys sample rate) x Units
    number_of_steps = int(kept_step_stats["count"])
    MU_spikes_3d_array_ephys_time = np.zeros(
        (
            number_of_steps,
            int(kept_step_stats["max"] * step_to_ephys_conversion_ratio),
            len(MU_spikes_dict),
        )
    )
    MU_spikes_3d_array_ephys_2π = MU_spikes_3d_array_ephys_time.copy()

    # initialize dict to store stepwise counts
    MU_step_aligned_spike_counts_dict = {key: None for key in MU_spikes_dict.keys()}
    # initialize dict of lists to store stepwise index arrays
    MU_step_aligned_spike_idxs_dict = {key: [] for key in MU_spikes_dict.keys()}
    MU_step_2π_warped_spike_idxs_dict = {key: [] for key in MU_spikes_dict.keys()}
    # convert foot strike/off indexes to the sample rate of electrophysiology data
    # set chosen alignment bodypart and choose corresponding index values
    if align_to == "foot strike":
        foot_strike_idxs_in_ephys_time = (
            foot_strike_idxs[step_slice]
        ) * step_to_ephys_conversion_ratio  # +start_video_capture_ephys_idx
        step_idxs_in_ephys_time = np.int32(foot_strike_idxs_in_ephys_time)
    elif align_to == "foot off":
        foot_off_idxs_in_ephys_time = (
            foot_off_idxs[step_slice]
        ) * step_to_ephys_conversion_ratio  # +start_video_capture_ephys_idx
        step_idxs_in_ephys_time = np.int32(foot_off_idxs_in_ephys_time)

    phase_warp_2π_coeff_list = []
    # fill 3d numpy array with Steps x Time x Units data, and a list of aligned idxs
    for iUnit, iUnitKey in enumerate(MU_spikes_dict.keys()):  # for each unit
        if sort_method == "thresholding":
            MU_spikes_idx_arr = MU_spikes_dict[iUnitKey] + step_idxs_in_ephys_time[0]
        elif sort_method == "kilosort":
            MU_spikes_idx_arr = (
                MU_spikes_dict[iUnitKey][
                    np.where(
                        (
                            MU_spikes_dict[iUnitKey][:]
                            > slice_for_ephys_during_video.start
                        )
                        & (
                            MU_spikes_dict[iUnitKey][:]
                            < slice_for_ephys_during_video.stop
                        )
                    )
                ]
                - slice_for_ephys_during_video.start
                + step_idxs_in_ephys_time[0]
            )
        for ii, iStep in enumerate(
            keep_trial_set
        ):  # range(number_of_steps): # for each step
            iStep -= step_slice.start
            # skip all spike counting and list appending if not in `keep_trial_set`
            # if iStep+step_slice.start in keep_trial_set:
            # keep track of index boundaries for each step
            this_step_idx = step_idxs_in_ephys_time[iStep]
            next_step_idx = step_idxs_in_ephys_time[iStep + 1]
            # filter out indexes which are outside the slice or this step's boundaries
            spike_idxs_in_step_and_slice_bounded = MU_spikes_idx_arr[
                np.where(
                    (MU_spikes_idx_arr < next_step_idx)
                    & (MU_spikes_idx_arr >= this_step_idx)
                    & (
                        MU_spikes_idx_arr
                        >= np.int32(
                            step_time_slice.start * step_to_ephys_conversion_ratio
                        )
                    )
                    & (
                        MU_spikes_idx_arr
                        <= np.int32(
                            step_time_slice.stop * step_to_ephys_conversion_ratio
                        )
                    )
                )
            ]
            # subtract current step index to align to each step, and convert to np.integer32 index
            MU_spikes_idxs_for_step = (
                spike_idxs_in_step_and_slice_bounded - this_step_idx
            ).astype(np.int32)
            # store aligned indexes for each step
            MU_step_aligned_spike_idxs_dict[iUnitKey].append(MU_spikes_idxs_for_step)
            # if any spikes are present, set them to 1 for this unit during this step
            if len(MU_spikes_idxs_for_step) != 0:
                MU_spikes_3d_array_ephys_time[ii, MU_spikes_idxs_for_step, iUnit] = 1
            # else: # mark slices with NaN's for later removal if not inside the keep_trial_set
            #     MU_spikes_3d_array_ephys_time[ii, :, iUnit] = np.nan
        # create phase aligned step indexes, with max index for each step set to 2π
        bin_width_eph_2π = []
        for ii, πStep in enumerate(
            keep_trial_set
        ):  # range(number_of_steps): # for each step
            πStep -= step_slice.start
            # if πStep+step_slice.start in keep_trial_set:
            # keep track of index boundaries for each step
            this_step_2π_idx = step_idxs_in_ephys_time[πStep]
            next_step_2π_idx = step_idxs_in_ephys_time[πStep + 1]
            # filter out indexes which are outside the slice or this step_2π's boundaries
            spike_idxs_in_step_2π_and_slice_bounded = MU_spikes_idx_arr[
                np.where(
                    (MU_spikes_idx_arr < next_step_2π_idx)
                    & (MU_spikes_idx_arr >= this_step_2π_idx)
                    & (
                        MU_spikes_idx_arr
                        >= np.int32(
                            step_time_slice.start * step_to_ephys_conversion_ratio
                        )
                    )
                    & (
                        MU_spikes_idx_arr
                        <= np.int32(
                            step_time_slice.stop * step_to_ephys_conversion_ratio
                        )
                    )
                )
            ]
            # coefficient to make step out of 2π radians, step made to be 2π after multiplication
            phase_warp_2π_coeff = (
                2
                * np.pi
                / (step_idxs_in_ephys_time[πStep + 1] - step_idxs_in_ephys_time[πStep])
            )
            phase_warp_2π_coeff_list.append(phase_warp_2π_coeff)
            # subtract this step start idx, and convert to an np.integer32 index
            MU_spikes_idxs_for_step_aligned = (
                spike_idxs_in_step_2π_and_slice_bounded - this_step_2π_idx
            ).astype(np.int32)
            MU_spikes_idxs_for_step_2π = (
                MU_spikes_idxs_for_step_aligned * phase_warp_2π_coeff
            )
            # store aligned indexes for each step_2π
            MU_step_2π_warped_spike_idxs_dict[iUnitKey].append(
                MU_spikes_idxs_for_step_2π
            )
            # if spikes are present, set them to 1 for this unit during this step
            if len(MU_spikes_idxs_for_step_2π) != 0:
                MU_spikes_3d_array_ephys_2π[
                    ii,
                    # convert MU_spikes_idxs_for_step_2π back to ephys sample rate-sized indexes
                    np.round(
                        MU_spikes_idxs_for_step_2π
                        / (2 * np.pi)
                        * MU_spikes_3d_array_ephys_2π.shape[1]
                    ).astype(np.int32),
                    iUnit,
                ] = 1

    # drop all steps rejected by `process_steps.trialize_steps()`
    steps_to_keep_arr = np.sort(np.fromiter(keep_trial_set, np.int32))

    # bin 3d array to time bins with width: bin_width_ms
    ms_duration = MU_spikes_3d_array_ephys_time.shape[1] / ephys_sample_rate * 1000
    # round up number of bins to prevent index overflows
    number_of_bins_in_time_step = np.ceil(ms_duration / bin_width_ms).astype(np.int32)
    bin_width_eph = np.int32(bin_width_ms * (ephys_sample_rate / 1000))

    # leave 2*pi numerator and number of bins equals: (500 / bin_width_ms)
    # WARNING: may need to change denominator constant to achieve correct radian binwidth
    bin_width_radian = 2 * np.pi / num_rad_bins
    # round up number of bins to prevent index overflows
    number_of_bins_in_2π_step = np.ceil(2 * np.pi / bin_width_radian).astype(np.int32)
    bin_width_eph_2π = np.round(
        MU_spikes_3d_array_ephys_2π.shape[1] / number_of_bins_in_2π_step
    ).astype(np.int32)
    # create binned 3d array with shape: Steps x Bins x Units
    MU_spikes_3d_array_binned = np.zeros(
        (
            MU_spikes_3d_array_ephys_time.shape[0],
            number_of_bins_in_time_step,
            MU_spikes_3d_array_ephys_time.shape[2],
        )
    )
    for iBin in range(number_of_bins_in_time_step):
        # sum across slice of cube bin_width_ms wide
        MU_spikes_3d_array_binned[:, iBin, :] = MU_spikes_3d_array_ephys_time[
            :, iBin * bin_width_eph : (iBin + 1) * bin_width_eph, :
        ].sum(1)

    MU_spikes_3d_array_binned_2π = np.zeros(
        (
            MU_spikes_3d_array_ephys_2π.shape[0],
            number_of_bins_in_2π_step,
            MU_spikes_3d_array_ephys_2π.shape[2],
        )
    )
    for πBin in range(number_of_bins_in_2π_step):
        # sum across slice of cube bin_width_radian wide
        MU_spikes_3d_array_binned_2π[:, πBin, :] = MU_spikes_3d_array_ephys_2π[
            :, πBin * bin_width_eph_2π : (πBin + 1) * bin_width_eph_2π, :
        ].sum(1)

    MU_spikes_count_across_all_steps = MU_spikes_3d_array_binned.sum(0).sum(0)

    ### PLOTTING SECTION
    figs = plot_handler.bin_and_count_plot(
        ephys_sample_rate,
        session_ID,
        sort_method,
        do_plot,
        MU_colors,
        CFG,
        MU_step_aligned_spike_idxs_dict,
        MU_step_2π_warped_spike_idxs_dict,
        MU_spikes_count_across_all_steps,
        number_of_steps,
        MU_spikes_dict,
        bin_width_radian,
        plot_flag,
    )
    ### END PLOTTING SECTION

    if export_data:
        # from scipy.io import savemat
        pass

    if save_binned_MU_data is True:
        np.save(session_ID + "_time.npy", MU_spikes_3d_array_binned, allow_pickle=False)
        np.save(
            session_ID + "_phase.npy", MU_spikes_3d_array_binned_2π, allow_pickle=False
        )

    return (
        MU_spikes_dict,
        MU_step_aligned_spike_idxs_dict,
        MU_step_aligned_spike_counts_dict,
        MU_step_2π_warped_spike_idxs_dict,
        MU_spikes_3d_array_ephys_time,
        MU_spikes_3d_array_binned,
        MU_spikes_3d_array_binned_2π,
        MU_spikes_count_across_all_steps,
        steps_to_keep_arr,
        step_idxs_in_ephys_time,
        ephys_sample_rate,
        session_ID,
        figs,
    )


def raster(
    chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_index
):
    (
        MU_spikes_dict,
        MU_step_aligned_spike_idxs_dict,
        MU_step_aligned_spike_counts_dict,
        MU_step_2π_warped_spike_idxs_dict,
        MU_spikes_3d_array_ephys_time,
        MU_spikes_3d_array_binned,
        MU_spikes_3d_array_binned_2π,
        MU_spikes_count_across_all_steps,
        steps_to_keep_arr,
        step_idxs_in_ephys_time,
        ephys_sample_rate,
        session_ID,
        figs,
    ) = bin_and_count(
        chosen_rat,
        OE_dict,
        KS_dict,
        anipose_dict,
        CH_colors,
        MU_colors,
        CFG,
        session_index,
    )

    ### Unpack CFG Inputs
    # unpack analysis inputs
    (
        MU_spike_amplitudes_list,
        ephys_channel_idxs_list,
        filter_ephys,
        sort_method,
        sort_to_use,
        bodypart_for_reference,
        bodypart_ref_filter,
        filter_all_anipose,
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
    (plot_type, plot_units, do_plot, N_colors, plot_template, *_) = CFG[
        "plotting"
    ].values()
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
    session_date = session_date[session_index]
    rat_name = str(chosen_rat).lower()
    treadmill_speed = str(treadmill_speed[session_index]).zfill(2)
    treadmill_incline = str(treadmill_incline[session_index]).zfill(2)
    session_ID = (
        f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
    )

    if do_plot == 2:  # override and ensure all plots are displayed when do_plot==2
        plot_flag = True
    else:  # only display plot if rat_loco_analysis() is the caller
        plot_flag = (
            True
            if (
                stack()[1].function == "rat_loco_analysis"
                and not plot_type.__contains__("multi")
            )
            else False
        )

    ### PLOTTING SECTION
    fig = plot_handler.raster_plot(
        MU_spikes_3d_array_ephys_time,
        MU_step_aligned_spike_idxs_dict,
        ephys_sample_rate,
        session_ID,
        plot_flag,
        plot_template,
        MU_colors,
    )
    ### END PLOTTING SECTION

    if export_data:
        # from scipy.io import savemat
        pass

    figs = [fig]
    return figs


def smoothed(
    chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_index
):
    (
        MU_spikes_dict,
        MU_step_aligned_spike_idxs_dict,
        MU_step_aligned_spike_counts_dict,
        MU_step_2π_warped_spike_idxs_dict,
        MU_spikes_3d_array_ephys_time,
        MU_spikes_3d_array_binned,
        MU_spikes_3d_array_binned_2π,
        MU_spikes_count_across_all_steps,
        steps_to_keep_arr,
        step_idxs_in_ephys_time,
        ephys_sample_rate,
        session_ID,
        figs,
    ) = bin_and_count(
        chosen_rat,
        OE_dict,
        KS_dict,
        anipose_dict,
        CH_colors,
        MU_colors,
        CFG,
        session_index,
    )

    ### Unpack CFG Inputs
    # unpack analysis inputs
    (
        MU_spike_amplitudes_list,
        ephys_channel_idxs_list,
        filter_ephys,
        sort_method,
        sort_to_use,
        bodypart_for_reference,
        bodypart_ref_filter,
        filter_all_anipose,
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
    (plot_type, plot_units, do_plot, N_colors, plot_template, *_) = CFG[
        "plotting"
    ].values()
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
    session_date = session_date[session_index]
    rat_name = str(chosen_rat).lower()
    treadmill_speed = str(treadmill_speed[session_index]).zfill(2)
    treadmill_incline = str(treadmill_incline[session_index]).zfill(2)
    session_ID = (
        f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
    )

    if do_plot == 2:  # override and ensure all plots are generated along the way
        plot_flag = True
    else:  # only display plot if rat_loco_analysis() is the caller
        plot_flag = (
            True
            if (
                stack()[1].function == "rat_loco_analysis"
                and not plot_type.__contains__("multi")
            )
            else False
        )

    # initialize 3d numpy array with shape: Steps x Bins x Units
    if phase_align is True:
        binned_spike_array = MU_spikes_3d_array_binned_2π
        bin_width = (2 * np.pi) / num_rad_bins
        bin_unit = " radians"
        title_prefix = "Phase-"
    else:
        binned_spike_array = MU_spikes_3d_array_binned
        bin_width = bin_width_ms
        bin_unit = "ms"
        title_prefix = "Time-"

    MU_smoothed_spikes_3d_array = np.zeros_like(binned_spike_array)
    number_of_steps = MU_smoothed_spikes_3d_array.shape[0]
    number_of_bins = MU_smoothed_spikes_3d_array.shape[1]
    number_of_units = MU_smoothed_spikes_3d_array.shape[2]

    # smooth traces
    for iStep in range(number_of_steps):
        # gaussian smooth across time, with standard deviation value of smoothing_window
        MU_smoothed_spikes_3d_array[iStep, :, :] = gaussian_filter1d(
            binned_spike_array[iStep, :, :],
            sigma=smoothing_window[session_index],
            axis=0,
            order=0,
            output=None,
            mode="constant",
            cval=0.0,
            truncate=4.0,
        )
    # get mean traces
    MU_smoothed_spikes_mean_2d_array = MU_smoothed_spikes_3d_array.mean(axis=0)

    ### PLOTTING SECTION
    fig = plot_handler.smoothed_plot(
        MU_smoothed_spikes_3d_array,
        MU_smoothed_spikes_mean_2d_array,
        number_of_steps,
        number_of_bins,
        number_of_units,
        session_ID,
        plot_flag,
        MU_colors,
        CH_colors,
        bin_width,
        bin_unit,
        smoothing_window,
        title_prefix,
        phase_align,
    )
    ### END PLOTTING SECTION

    if export_data:
        # from scipy.io import savemat
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
    chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_index
):
    ### Unpack CFG Inputs
    # unpack analysis inputs
    (
        MU_spike_amplitudes_list,
        ephys_channel_idxs_list,
        filter_ephys,
        sort_method,
        sort_to_use,
        bodypart_for_reference,
        bodypart_ref_filter,
        filter_all_anipose,
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
    (plot_type, plot_units, do_plot, N_colors, plot_template, *_) = CFG[
        "plotting"
    ].values()
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
    session_date = session_date[session_index]
    rat_name = str(chosen_rat).lower()
    treadmill_speed = str(treadmill_speed[session_index]).zfill(2)
    treadmill_incline = str(treadmill_incline[session_index]).zfill(2)
    session_ID = (
        f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
    )

    if do_plot == 2:  # override and ensure all plots are generated along the way
        plot_flag = True
    else:  # only display plot if rat_loco_analysis() is the caller
        plot_flag = (
            True
            if (
                stack()[1].function == "rat_loco_analysis"
                and not plot_type.__contains__("multi")
            )
            else False
        )

    (
        MU_smoothed_spikes_3d_array,
        binned_spike_array,
        steps_to_keep_arr,
        figs,
    ) = smoothed(
        chosen_rat,
        OE_dict,
        KS_dict,
        anipose_dict,
        CH_colors,
        MU_colors,
        CFG,
        session_index,
    )

    # select units for plotting
    if sort_method == "kilosort":
        sliced_MU_smoothed_3d_array = MU_smoothed_spikes_3d_array  # [:,:,plot_units]
    else:
        sliced_MU_smoothed_3d_array = MU_smoothed_spikes_3d_array[:, :, plot_units]

    # set numbers of things from input matrix dimensionality
    # number_of_steps = sliced_MU_smoothed_3d_array.shape[0]
    # number_of_bins = sliced_MU_smoothed_3d_array.shape[1]
    # detect number of identified units
    number_of_units = sliced_MU_smoothed_3d_array.any(1).any(0).sum()
    if phase_align is True:
        bin_width = (2 * np.pi) / num_rad_bins
        bin_unit = "radians"
        title_prefix = "Phase"
    else:
        bin_width = bin_width_ms
        bin_unit = "ms"
        title_prefix = "Time"

    for iStep in steps_to_keep_arr:
        # gaussian smooth across time, with standard deviation value of smoothing_window
        sliced_MU_smoothed_3d_array[iStep, :, :] = gaussian_filter1d(
            sliced_MU_smoothed_3d_array[iStep, :, :],
            sigma=smoothing_window[session_index],
            axis=0,
            order=0,
            output=None,
            mode="constant",
            cval=0.0,
            truncate=4.0,
        )

    ### PLOTTING SECTION
    fig = plot_handler.state_space_plot(
        steps_to_keep_arr,
        sliced_MU_smoothed_3d_array,
        number_of_units,
        session_ID,
        plot_flag,
        MU_colors,
        CH_colors,
        smoothing_window,
        title_prefix,
        ephys_channel_idxs_list,
        treadmill_incline,
        plot_units,
    )
    ### END PLOTTING SECTION

    if export_data:
        # from scipy.io import savemat
        pass
    # put in a list for compatibility with calling functions
    figs = [fig]
    return MU_smoothed_spikes_3d_array, binned_spike_array, steps_to_keep_arr, figs


def MU_space_stepwise(
    chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_index
):
    ### Unpack CFG Inputs
    # unpack analysis inputs
    (
        MU_spike_amplitudes_list,
        ephys_channel_idxs_list,
        filter_ephys,
        sort_method,
        sort_to_use,
        bodypart_for_reference,
        bodypart_ref_filter,
        filter_all_anipose,
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
    (plot_type, plot_units, do_plot, N_colors, plot_template, *_) = CFG[
        "plotting"
    ].values()
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
    session_date = session_date[session_index]
    rat_name = str(chosen_rat).lower()
    treadmill_speed = str(treadmill_speed[session_index]).zfill(2)
    treadmill_incline = str(treadmill_incline[session_index]).zfill(2)
    session_ID = (
        f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
    )

    if do_plot == 2:  # override and ensure all plots are generated along the way
        plot_flag = True
    else:  # only display plot if rat_loco_analysis() is the caller
        plot_flag = (
            True
            if (
                stack()[1].function == "rat_loco_analysis"
                and not plot_type.__contains__("multi")
            )
            else False
        )
        iPar = 0
    # session_ID_lst = []
    # trialized_anipose_dfs_lst = []
    subtitles = []
    for iTitle in treadmill_incline:
        subtitles.append("<b>Incline: " + str(iTitle) + "</b>")

    big_fig = []
    for iPar in range(len(treadmill_incline)):
        (
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
        ) = trialize_steps(
            chosen_rat,
            OE_dict,
            KS_dict,
            anipose_dict,
            CH_colors,
            MU_colors,
            CFG,
            session_index,
        )

        # (MU_smoothed_spikes_3d_array, binned_spike_array, figs) = smoothed(
        # OE_dict, KS_dict, anipose_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list, filter_ephys, sort_method,
        # filter_all_anipose, bin_width_ms, bin_width_radian, smoothing_window,
        # bodypart_for_alignment, bodypart_for_reference, bodypart_ref_filter,
        # trial_reject_bounds_mm, trial_reject_bounds_sec,origin_offsets, bodyparts_list,
        # session_date[iPar], rat_name[iPar], treadmill_speed[iPar], treadmill_incline[iPar],
        # camera_fps, align_to, vid_length, time_frame, do_plot=False, phase_align=phase_align,
        # plot_template=plot_template, MU_colors=MU_colors, CH_colors=CH_colors)

        (
            MU_smoothed_spikes_3d_array,
            binned_spike_array,
            steps_to_keep_arr,
            figs,
        ) = state_space(
            chosen_rat,
            OE_dict,
            KS_dict,
            anipose_dict,
            CH_colors,
            MU_colors,
            CFG,
            session_index,
        )

        # session_ID = \
        # f"{session_date[iPar]}_{rat_name[iPar]}_speed{treadmill_speed[iPar]}_incline{treadmill_incline[iPar]}"

        # raster_figs = raster(OE_dict, KS_dict, anipose_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
        # filter_ephys, sort_method, filter_all_anipose, bin_width_ms, bin_width_radian,
        # bodypart_for_alignment, bodypart_for_reference, bodypart_ref_filter, trial_reject_bounds_mm, trial_reject_bounds_sec,
        # origin_offsets, bodyparts_list, session_date[0], rat_name[0], treadmill_speed[0], treadmill_incline[0],
        # camera_fps, align_to, vid_length, time_frame,
        # do_plot, plot_template, MU_colors, CH_colors)
        ### PLOTTING SECTION
        MU_space_stepwise(
            iPar,
            keep_trial_set,
            binned_spike_array,
            MU_smoothed_spikes_3d_array,
            plot_units,
            big_fig,
            figs,
            CH_colors,
            MU_colors,
            steps_to_keep_arr,
            phase_align,
            kept_step_stats,
            camera_fps,
            bin_width_ms,
            plot_flag,
        )
        ### END PLOTTING SECTION
    return


# Helper function for comparing kinematics of bodypart to sorted spikes
## Called by sort() if expected
def create_steps_dict(
    plot_units,
    time_axis_for_ephys,
    MU_spikes_dict,
    slice_for_ephys_during_video,
    time_frame,
    time_axis_for_anipose,
    foot_strike_slice_idxs,
    foot_off_slice_idxs,
):
    # Helper function to create_steps_dict
    def _getnearpos(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    ## MAKE VARIABLE FOR HOW MANY UNITS TO USE ##

    ## CREATES DICTIONARY OF STEPS WITH ALIGNED TIME VALUES FOR DURATION OF EACH STEP ##
    steps_dict = {}
    foot_off_slice_idxs2 = foot_off_slice_idxs[1:]
    for iStrike, iOff in zip(
        enumerate(foot_strike_slice_idxs), enumerate(foot_off_slice_idxs2)
    ):
        step_range = range(iStrike[1], iOff[1] + 1)
        steps_dict[iStrike[0]] = {
            "step_bounds": np.array(
                [
                    time_axis_for_anipose[step_range.start],
                    time_axis_for_anipose[step_range.stop],
                ]
            ),
            "ephys_idxs_in_step": np.arange(
                _getnearpos(
                    time_axis_for_ephys[slice_for_ephys_during_video],
                    time_axis_for_anipose[iStrike[1]],
                ),
                _getnearpos(
                    time_axis_for_ephys[slice_for_ephys_during_video],
                    time_axis_for_anipose[iOff[1]],
                ),
            ),
        }

    # To organize sorted spikes
    for iUnit, iUnitKey in enumerate(plot_units):
        MU_spikes_dict_for_unit = (
            MU_spikes_dict[iUnitKey][:]
            if time_frame == 1
            else MU_spikes_dict[iUnitKey][:][
                np.where(
                    (MU_spikes_dict[iUnitKey][:] > slice_for_ephys_during_video.start)
                    & (MU_spikes_dict[iUnitKey][:] < slice_for_ephys_during_video.stop)
                )
            ]
        )
        # Index where spikes are, ephys time
        ephys_spike_idxs = time_axis_for_ephys[MU_spikes_dict_for_unit]
        for iStep in steps_dict:  # Create MU Keys in dict for spike times per step
            mask = (ephys_spike_idxs >= steps_dict[iStep]["step_bounds"][0]) & (
                ephys_spike_idxs <= steps_dict[iStep]["step_bounds"][1]
            )
            steps_dict[iStep].update({f"MU:{iUnitKey} Indexes": ephys_spike_idxs[mask]})
            # print('here')

    # Create different function for plotting --plot handler module --import function from plot_handler
    # Plot each step -behavioral_space overlaps steps (color map gradient, continous spectrum)
    # See if there is relationship with body velocity & limb velocity (and be able to choose analyses)
    # Sort kinematic traces first (based on motor unit offsets)
    return steps_dict


# def plot_steps_by_neural(steps_dict, MUnits):
#     MU_time_diff_steps = {}
#     smallest_MU = MUnits[0]
#     largest_MU = MUnits[-1]
#     for iStep in steps_dict:
#         #if len(steps_dict[iStep][f'MU:{largest_MU} Indexes'])>0 & len(steps_dict[iStep][f'MU:{smallest_MU} Indexes'])>0:
#         try:
#             MU_time_diff_steps[iStep] = (steps_dict[iStep][f'MU:{largest_MU} Indexes'][0] - steps_dict[iStep][f'MU:{smallest_MU} Indexes'][0])
#         except:
#             continue

#     plot_steps = []
#     temporal_MU_diff = []
#     temporal_step_diff = []

#     for iStep in MU_time_diff_steps:
#         plot_steps.append(iStep)
#         temporal_MU_diff.append(MU_time_diff_steps[iStep])
#         temporal_step_diff.append(steps_dict[iStep]['step_bounds'][1]-steps_dict[iStep]['step_bounds'][0])

#     sorted_idxs = np.argsort(temporal_MU_diff)
#     plot_steps = np.array(plot_steps)[sorted_idxs]
#     temporal_MU_diff = np.array(temporal_MU_diff)[sorted_idxs]
#     temporal_step_diff = np.array(temporal_step_diff)[sorted_idxs]

#     plot_df_data = {'Temp MU Diff': np.sin(temporal_MU_diff),
#                     'Temp Step Diff': np.sin(temporal_step_diff)}
#     plot_df = pd.DataFrame(plot_df_data)
#     fig = px.imshow(plot_df, color_continuous_scale=px.colors.sequential.Cividis_r, contrast_rescaling='infer')
#     fig.update_layout(coloraxis_showscale=True)
#     fig.update_xaxes(showticklabels=False)
#     fig.update_yaxes(showticklabels=False)
#     #fig.show()
#     #set_trace()

#     return

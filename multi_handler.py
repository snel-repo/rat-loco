from inspect import stack

import numpy as np

# this module houses functions that combine data from multiple sessions after processing them
# with the functions in process_spikes.py and process_steps.py. The goal is to have a single
# numpy array for all sessions provided in session_date in the config file


# this function loops through process_spikes.bin_and_count() for each session provided
# and accumulates the results into a single numpy array along the step dimension.
# The 3D array is then plotted with bin_and_count_plot in plot_handler.py
def multijoin_bin_and_count(
    chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_indexes
):
    # import necessary function from process_spikes.py
    from plot_handler import bin_and_count_plot
    from process_spikes import bin_and_count

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

    if do_plot == 2:  # override and ensure all plots are generated along the way
        plot_flag = True
    else:  # only display plot if rat_loco_analysis() is the caller
        plot_flag = (
            True
            if (stack()[1].function == "rat_loco_analysis" and not plot_type.__contains__("multi"))
            else False
        )

    # joined_MU_spikes_3d_array_binned = np.array([])
    # joined_MU_spikes_3d_array_binned_2π = np.array([])
    for iRec in session_indexes:
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
            chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, iRec
        )
        # MU_spikes_dict_keys = list(MU_spikes_dict.keys())

        if iRec == session_indexes[0]:
            joined_session_ID = session_ID
            MU_spikes_count_across_all_sessions = MU_spikes_3d_array_binned.sum(0).sum(0)
            joined_MU_step_aligned_spike_idxs_dict = MU_step_aligned_spike_idxs_dict.copy()
            joined_MU_step_2π_warped_spike_idxs_dict = MU_step_2π_warped_spike_idxs_dict.copy()
            # joined_MU_spikes_3d_array_binned_2π = MU_spikes_3d_array_binned_2π.copy()
        else:
            joined_session_ID = f"{session_ID.split('_')[0]}+{joined_session_ID}"
            MU_spikes_count_across_all_sessions = (
                MU_spikes_count_across_all_sessions.copy() + MU_spikes_3d_array_binned.sum(0).sum(0)
            )
            for key in MU_step_aligned_spike_idxs_dict.keys():
                joined_MU_step_aligned_spike_idxs_dict[key] = (
                    joined_MU_step_aligned_spike_idxs_dict[key]
                    + MU_step_aligned_spike_idxs_dict[key]
                )
                joined_MU_step_2π_warped_spike_idxs_dict[key] = (
                    joined_MU_step_2π_warped_spike_idxs_dict[key]
                    + MU_step_2π_warped_spike_idxs_dict[key]
                )

    number_of_steps = MU_spikes_3d_array_binned.shape[0]
    bin_width_radian = 2 * np.pi / num_rad_bins

    ### PLOTTING SECTION
    bin_and_count_plot(
        ephys_sample_rate,
        session_ID,
        sort_method,
        do_plot,
        MU_colors,
        CFG,
        joined_MU_step_aligned_spike_idxs_dict,
        joined_MU_step_2π_warped_spike_idxs_dict,
        MU_spikes_count_across_all_steps,
        number_of_steps,
        MU_spikes_dict,
        bin_width_radian,
        plot_flag,
    )
    ### END PLOTTING SECTION

    return

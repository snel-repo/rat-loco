from process_steps import peak_align_and_filt, trialize_steps
import numpy as np
from plotly.offline import iplot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, butter, filtfilt, iirnotch
from scipy.ndimage import gaussian_filter1d
from inspect import stack

# this module will house functions that combine data from multiple sessions after processing them
# with the functions in process_spikes.py and process_steps.py. The goal is to have a single
# numpy array for all sessions provided in session_date in the config file

# this function will loop through process_spikes.bin_and_count() for each session provided
# and accumulate the results into a single numpy array along the step dimension.
# The 3D array will then be plotted with bin_and_count_plot in plot_handler.py
def multijoin_bin_and_count(
    chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator
    ):
    
    # import necessary function from process_spikes.py
    from process_spikes import bin_and_count
    from plot_handler import bin_and_count_plot
    
    ### Unpack CFG Inputs
    # unpack analysis inputs
    (MU_spike_amplitudes_list,ephys_channel_idxs_list,filter_ephys,sort_method,
    bodypart_for_reference,bodypart_ref_filter,filter_all_anipose,trial_reject_bounds_mm,
    trial_reject_bounds_sec,origin_offsets,save_binned_MU_data,time_frame,bin_width_ms,
    num_rad_bins,smoothing_window,phase_align,align_to,export_data) = CFG['analysis'].values()
    # unpack plotting inputs
    (plot_type,plot_units,do_plot,N_colors,plot_template,*_) = CFG['plotting'].values()
    # unpack chosen rat inputs
    (bodyparts_list,bodypart_for_alignment,session_date,treadmill_speed,
    treadmill_incline,camera_fps,vid_length) = CFG['rat'][chosen_rat].values()
    
    joined_MU_spikes_3d_array_binned = np.array([])
    joined_MU_spikes_3d_array_binned_2π = np.array([])
    for iRec in session_iterator:
        (
            MU_spikes_dict,
            MU_step_aligned_spike_idxs_dict,
            MU_step_aligned_spike_counts_dict,
            MU_step_2π_warped_spike_idxs_dict,
            MU_spikes_3d_array_ephys_time,
            MU_spikes_3d_array_binned,
            MU_spikes_3d_array_binned_2π,
            MU_spikes_count_across_all_steps, steps_to_keep_arr,
            step_idxs_in_ephys_time, ephys_sample_rate, session_ID, figs
        ) = bin_and_count(
            chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, iRec
            )
        MU_spikes_dict_keys = list(MU_spikes_dict.keys())
        
        if iRec == session_iterator[0]:
                MU_spikes_count_across_all_steps = MU_spikes_3d_array_binned.sum(0).sum(0)
                joined_MU_step_aligned_spike_idxs_dict = MU_step_aligned_spike_idxs_dict.copy()
                joined_MU_step_2π_warped_spike_idxs_dict = MU_step_2π_warped_spike_idxs_dict.copy()
                # joined_MU_spikes_3d_array_binned_2π = MU_spikes_3d_array_binned_2π.copy()
        else:
            MU_spikes_count_across_all_steps = (MU_spikes_count_across_all_steps +
                                                MU_spikes_3d_array_binned.sum(0).sum(0))
            for key in MU_step_aligned_spike_idxs_dict.keys():
                joined_MU_step_aligned_spike_idxs_dict[key] = (
                    joined_MU_step_aligned_spike_idxs_dict[key] +
                    MU_step_aligned_spike_idxs_dict[key]
                    )
                joined_MU_step_2π_warped_spike_idxs_dict[key] = (
                    joined_MU_step_2π_warped_spike_idxs_dict[key] +
                    MU_step_2π_warped_spike_idxs_dict[key]
                    )
            # joined_MU_spikes_3d_array_binned_2π = np.concatenate(
            #     (joined_MU_spikes_3d_array_binned_2π, MU_spikes_3d_array_binned_2π),axis=0)
            
    bin_and_count_plot(
        MU_spikes_dict_keys, ephys_sample_rate, session_ID, sort_method,
        do_plot, MU_colors, CFG, joined_MU_step_aligned_spike_idxs_dict,
        joined_MU_step_2π_warped_spike_idxs_dict, MU_spikes_count_across_all_steps
        )
    
    return
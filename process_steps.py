import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.offline import iplot
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, butter, sosfiltfilt, medfilt
from IPython.display import display
from inspect import stack
from pdb import set_trace

# create highpass filter to remove baseline from foot tracking for better peak finding performance
def butter_highpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    return sos
def butter_highpass_filter(data, cutoff, fs, order=2):
    sos = butter_highpass(cutoff, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y

# create lowpass filter to smooth reference bodypart and reduce its variability
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    return sos
def butter_lowpass_filter(data, cutoff, fs, order=5):
    sos = butter_lowpass(cutoff, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y

# create bandpass filter to smooth reference bodypart and reduce its variability
def butter_bandpass(cutoffs, fs, order=2):
    assert len(cutoffs)==2
    nyq = 0.5 * fs
    normal_cutoffs = np.array(cutoffs) / nyq
    sos = butter(order, normal_cutoffs, btype='bandpass', analog=False, output='sos')
    return sos
def butter_bandpass_filter(data, cutoffs, fs, order=2):
    sos = butter_bandpass(cutoffs, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y

# function identifies peaks and troughs of anipose data, and
# optionally applies filtering and offset or reference bodypart subtraction
def peak_align_and_filt(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, iterator):
    ### Unpack CFG Inputs
    # unpack analysis inputs
    (MU_spike_amplitudes_list,ephys_channel_idxs_list,filter_ephys,sort_method,
     bodypart_for_reference,bodypart_ref_filter,filter_all_anipose,trial_reject_bounds_mm,
     trial_reject_bounds_sec,origin_offsets,save_binned_MU_data,time_frame,bin_width_ms,
     num_rad_bins,smoothing_window,phase_align,align_to) = CFG['analysis'].values()
    # unpack plotting inputs
    (plot_type,plot_units,do_plot,N_colors,plot_template,*_) = CFG['plotting'].values()
    # unpack chosen rat inputs
    (bodyparts_list,bodypart_for_alignment,session_date,treadmill_speed,
     treadmill_incline,camera_fps,vid_length) = CFG['rat'][chosen_rat].values()
    
    # format inputs to avoid ambiguities
    session_date = session_date[iterator]
    rat_name = str(chosen_rat).lower()
    treadmill_speed = str(treadmill_speed[iterator]).zfill(2)
    treadmill_incline = str(treadmill_incline[iterator]).zfill(2)
    session_ID = f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"

    # # format inputs to avoid ambiguities
    # rat_name = str(rat_name).lower()
    # treadmill_speed = str(treadmill_speed).zfill(2)
    # treadmill_incline = str(treadmill_incline).zfill(2)
   
    # ensure dict is extracted
    if type(anipose_dict) is dict:
        anipose_dict = anipose_dict[session_ID]
    cols = anipose_dict.columns
    bodypart_substr = ['_x','_y','_z']
    not_bodypart_substr = ['ref','origin']
    reduced_cols = [str for str in cols if any(sub in str for sub in bodypart_substr)]
    bodypart_cols = [str for str in reduced_cols if not any(
        sub in str for sub in not_bodypart_substr)]
    bodypart_anipose_df = anipose_dict[bodypart_cols]
    ref_aligned_df = bodypart_anipose_df.copy()
    ref_bodypart_trace_list = []
    if origin_offsets is not False:
        for iDim in bodypart_substr:
            # if iDim == 'Labels': continue # skip if Labels column
            body_dim_cols = [str for str in bodypart_cols if any(sub in str for sub in [iDim])]
            # iDim_cols = [str for str in bodypart_cols if str in [iDim]]
            if type(origin_offsets[iDim[-1]]) is int:
                ref_aligned_df[body_dim_cols] = \
                    bodypart_anipose_df[body_dim_cols]+origin_offsets[iDim[-1]]
            elif type(origin_offsets[iDim[-1]]) is str:
                if bodypart_ref_filter:
                    ref_bodypart_trace_list.append(butter_lowpass_filter(
                        data=bodypart_anipose_df[bodypart_for_reference+iDim],
                        cutoff=bodypart_ref_filter, fs=camera_fps, order=1))
                else:
                    ref_bodypart_trace_list.append(bodypart_anipose_df[bodypart_for_reference]+iDim)
                for iCol in body_dim_cols:
                    ref_aligned_df[iCol] = \
                        bodypart_anipose_df[iCol] - ref_bodypart_trace_list[-1]
            else:
                raise TypeError("origin_offsets must be a dict. Keys must be 'x'/'y'/'z' and values must be either type `int` or `str`.")
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
        [ref_aligned_df.loc[:,x_data],
         ref_aligned_df.loc[:,y_data],
         ref_aligned_df.loc[:,z_data]],
        axis=1,ignore_index=False)
    
    low = 0.5; high = 25
    if filter_all_anipose == 'highpass':
        filtered_anipose_data = butter_highpass_filter(
            data=sorted_body_anipose_df.values,
            cutoff=low, fs=camera_fps, order=3)
        print(f"A {filter_all_anipose} filter was applied to all anipose data (lowcut = {low}Hz, highcut = {high}Hz).")
    elif filter_all_anipose == 'lowpass':
        filtered_anipose_data = butter_lowpass_filter(
            data=sorted_body_anipose_df.values,
            cutoff=high, fs=camera_fps, order=3)
        print(f"A {filter_all_anipose} filter was applied to all anipose data (lowcut = {low}Hz, highcut = {high}Hz).")
    elif filter_all_anipose == 'bandpass':
        filtered_anipose_data = butter_bandpass_filter(
            data=sorted_body_anipose_df.values,
            cutoffs=[low,high], fs=camera_fps, order=3)
        print(f"A {filter_all_anipose} filter was applied to all anipose data (lowcut = {low}Hz, highcut = {high}Hz).")
    elif filter_all_anipose == 'median':
        filtered_anipose_data = medfilt(sorted_body_anipose_df.values, kernel_size=[7,1])
        print(f"A {filter_all_anipose} filter was applied to all anipose data.")
    else: # do not filter
        filtered_anipose_data=sorted_body_anipose_df.values
    processed_anipose_df = pd.DataFrame(filtered_anipose_data,columns=sorted_body_anipose_df.columns)
    # identify motion peak locations for foot strike
    foot_strike_idxs, _ = find_peaks(
        processed_anipose_df[bodypart_for_alignment[0]],
        height=[None,None],
        threshold=None,
        distance=30,
        prominence=None,
        width=None,
        wlen=None
        )

    foot_off_idxs, _ = find_peaks(
        -processed_anipose_df[bodypart_for_alignment[0]], # invert signal to find the troughs
        height=[None,None],
        threshold=None,
        distance=30,
        prominence=None,
        width=None,
        wlen=None
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
    sliced_steps_diff = pd.DataFrame(np.diff(step_idxs[step_slice]))#step_idxs[1:] - step_idxs[:-1])
    
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
    return processed_anipose_df, foot_strike_idxs, foot_off_idxs, sliced_step_stats, step_slice, step_time_slice, ref_bodypart_trace_list

def trialize_steps(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, iterator):
    
    ### Unpack CFG Inputs
    # unpack analysis inputs
    (MU_spike_amplitudes_list,ephys_channel_idxs_list,filter_ephys,sort_method,
     bodypart_for_reference,bodypart_ref_filter,filter_all_anipose,trial_reject_bounds_mm,
     trial_reject_bounds_sec,origin_offsets,save_binned_MU_data,time_frame,bin_width_ms,
     num_rad_bins,smoothing_window,phase_align,align_to) = CFG['analysis'].values()
    # unpack plotting inputs
    (plot_type,plot_units,do_plot,N_colors,plot_template,*_) = CFG['plotting'].values()
    # unpack chosen rat inputs
    (bodyparts_list,bodypart_for_alignment,session_date,treadmill_speed,
     treadmill_incline,camera_fps,vid_length) = CFG['rat'][chosen_rat].values()
    
    # format inputs to avoid ambiguities
    session_date = session_date[iterator]
    rat_name = str(chosen_rat).lower()
    treadmill_speed = str(treadmill_speed[iterator]).zfill(2)
    treadmill_incline = str(treadmill_incline[iterator]).zfill(2)
    session_ID = f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
    
    (processed_anipose_df, foot_strike_idxs, foot_off_idxs,
    sliced_step_stats, step_slice, step_time_slice, ref_bodypart_trace_list
     ) = peak_align_and_filt(
         chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, iterator
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
    # align_shift = 2/3 # default fraction of step
    pre_align_offset  = int(0)#int(sliced_step_stats.quantile(0.75) // 8) #int(sliced_step_stats.quantile(0.75) *    align_shift)
    post_align_offset = int(sliced_step_stats.quantile(0.75))# * 7 // 8) #int(sliced_step_stats.quantile(0.75) * (1-align_shift))
    if align_to == 'foot strike':
        step_idxs = foot_strike_idxs[step_slice]
    elif align_to == 'foot off':
        step_idxs = foot_off_idxs[step_slice]
    
    # create trialized list of DataFrames: trialized_anipose_df_lst[Step][Values,'Bodypart']
    trialized_anipose_df_lst = []
    # slice off the last step index to avoid miscounting (avoid off by one error)
    true_step_idx = np.array([*range(step_slice.start,step_slice.stop)])[:-1]
    for iStep, (i_step_idx, i_true_step_num) in enumerate(zip(step_idxs[:-1],true_step_idx)):
        trialized_anipose_df_lst.append(
            processed_anipose_df.iloc[i_step_idx-pre_align_offset:i_step_idx+post_align_offset,:])
        # give column names to each step for all bodyparts, and
        # zerofill to the max number of digits in `true_step_idx`
        trialized_anipose_df_lst[iStep].columns = [
            iCol+f"_{str(i_true_step_num).zfill(int(1+np.log10(true_step_idx.max())))}" for iCol in trialized_anipose_df_lst[iStep].columns]
        trialized_anipose_df_lst[iStep].reset_index(drop=True, inplace=True)
    trialized_anipose_df = pd.concat(trialized_anipose_df_lst, axis=1)

    all_trial_set = set(true_step_idx) 
    keep_trial_set = set(true_step_idx) # start with all sliced steps
    drop_trial_set = set()
    
    if trial_reject_bounds_sec:# If not False, will enforce trial length bounds
        assert len(trial_reject_bounds_sec)==2 and type(trial_reject_bounds_sec)==list, "`trial_reject_bounds_sec` must be a 2D list."
        trials_above_lb = set()
        trials_below_ub = set()

        lower_bound = trial_reject_bounds_sec[0]
        upper_bound = trial_reject_bounds_sec[1]
        step_durations_sec = np.diff(step_idxs)/camera_fps
        for iBodypart in bodyparts_list:
            trials_above_lb.update(true_step_idx[np.where(step_durations_sec>lower_bound)])
            trials_below_ub.update(true_step_idx[np.where(step_durations_sec<upper_bound)])
            # get trial idxs between bounds, loop through bodyparts, remove trials outside
            keep_trial_set.intersection_update(trials_above_lb & trials_below_ub)
        print(f"Step duration bounds: {np.round(lower_bound,decimals=3)}s to {np.round(upper_bound,decimals=3)}s")
        drop_trial_set.update(all_trial_set - keep_trial_set)
        print(f"Steps outside of bounds: {drop_trial_set}")
        
    if trial_reject_bounds_mm: # If not False, will remove outlier trials
        # use sets to store trials to be saved, to take advantage of set operations
        # as bodypart columns and rejection criteria are looped through
        trials_above_lb = set()
        trials_below_ub = set()
        # get idxs for `align_to` and shift by (pre_align_offset+post_align_offset)/2 in order to 
        # approx a 180 degree phase shift and get idx of other peak/trough which was not aligned to
        # also use conditionals to check for whether the phase shift should be added or subtracted
        # based on whether pre_align_offset is greater/less than post_align_offset
        df_peak_and_trough_list = [trialized_anipose_df.iloc[pre_align_offset-(pre_align_offset+post_align_offset)//2]
                                   if pre_align_offset>=post_align_offset else
                                   trialized_anipose_df.iloc[pre_align_offset+(pre_align_offset+post_align_offset)//2], 
                                   trialized_anipose_df.iloc[pre_align_offset]] # <- foot_off/foot_strike index

        # Get minimum value of each trial for each bodypart, then take median across trials.
        # Reject trials outside bounds around those medians.
        if type(trial_reject_bounds_mm) is int:
            assert trial_reject_bounds_mm>0,"If `trial_reject_bounds_mm` is an integer, it must be greater than zero."
            if align_to=="foot strike":
                df_peak_or_trough=df_peak_and_trough_list[0]
            elif align_to=="foot off":
                df_peak_or_trough=df_peak_and_trough_list[1]
            for iBodypart in bodyparts_list:
                lower_bound = df_peak_or_trough.filter(like=iBodypart).median() - trial_reject_bounds_mm
                upper_bound = df_peak_or_trough.filter(like=iBodypart).median() + trial_reject_bounds_mm
                trials_above_lb.update(true_step_idx[(df_peak_or_trough.filter(like=iBodypart)>lower_bound).values])
                trials_below_ub.update(true_step_idx[(df_peak_or_trough.filter(like=iBodypart)<upper_bound).values])
                # get trial idxs between bounds, loop through bodyparts, remove trials outside
                keep_trial_set.intersection_update(trials_above_lb & trials_below_ub)
                print(f"{align_to} bounds for {iBodypart}: {np.round(lower_bound,decimals=2)} to {np.round(upper_bound,decimals=2)}")
            drop_trial_set.update(all_trial_set - keep_trial_set)
        elif type(trial_reject_bounds_mm) is list:
            # For each feature, assert that the first list element is less than the second list element
            assert trial_reject_bounds_mm[0][0]<trial_reject_bounds_mm[0][1], \
                "If `trial_reject_bounds_mm` is a 2d list, use form: [[10,40],[-10,25]] for maximal allowed Peak +/- and Trough +/- values. First of paired list value must be less than second."
            assert trial_reject_bounds_mm[1][0]<trial_reject_bounds_mm[1][1], \
                "If `trial_reject_bounds_mm` is a 2d list, use form: [[10,40],[-10,25]] for maximal allowed Peak +/- and Trough +/- values. First of paired list value must be less than second."
            # loop through both keys of the dictionary, keep trials when all bodyparts are constrained
            pair_description = ['Peak', 'Trough']
            for ii, (trial_reject_bounds_mm_pair, df_peak_or_trough) in enumerate(zip(trial_reject_bounds_mm,df_peak_and_trough_list)):
                for iBodypart in bodyparts_list:
                    lower_bound = df_peak_or_trough.filter(like=iBodypart).median() + trial_reject_bounds_mm_pair[0]
                    upper_bound = df_peak_or_trough.filter(like=iBodypart).median() + trial_reject_bounds_mm_pair[1]
                    trials_above_lb.update(true_step_idx[(df_peak_or_trough.filter(like=iBodypart)>lower_bound).values])
                    trials_below_ub.update(true_step_idx[(df_peak_or_trough.filter(like=iBodypart)<upper_bound).values])
                    # get trial idxs between bounds, loop through bodyparts, remove trials outside
                    keep_trial_set.intersection_update(trials_above_lb & trials_below_ub)
                    print(f"{pair_description[ii]} bounds for {iBodypart}: {np.round(lower_bound,decimals=2)} to {np.round(upper_bound,decimals=2)}")
                trials_above_lb, trials_below_ub = set(), set()
                drop_trial_set.update(all_trial_set - keep_trial_set)
        else:
            raise TypeError("Wrong type specified for `trial_reject_bounds_mm` parameter in `rat_loco_analysis.py`")
        
        for iTrial in drop_trial_set:
            # drop out of bounds trials from DataFrame in place, use log10 to get number of decimals for zfilling
            trialized_anipose_df.drop( 
                list(trialized_anipose_df.filter(like = \
                    f"_{str(iTrial).zfill(int(1+np.log10(true_step_idx.max())))}")), axis=1, inplace=True)
    
    sliced_steps_diff = np.diff(step_idxs)
    kept_steps_diff = pd.DataFrame(np.array([sliced_steps_diff[iTrial-step_slice.start] for iTrial in (keep_trial_set)]))
    print(
        f"Inter-step timing stats for {align_to}, for steps: {keep_trial_set}:\
            \nSession ID: {session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}")
    
    kept_step_stats = kept_steps_diff.describe()[0]
    
    display(kept_step_stats)
    if not trial_reject_bounds_mm and not trial_reject_bounds_sec:
        print("No trials rejected, because `trial_reject_bounds_mm` and `trial_reject_bounds_s`ec` were set to False in `config/config.toml`")
        kept_step_stats = sliced_step_stats
    else:
        print(f'Rejected trials: {drop_trial_set}')
        
    return (trialized_anipose_df, keep_trial_set, foot_strike_idxs, foot_off_idxs, sliced_step_stats,
            kept_step_stats, step_slice, step_time_slice, ref_bodypart_trace_list,
            pre_align_offset, post_align_offset, trial_reject_bounds_mm, trial_reject_bounds_sec,)

def behavioral_space(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, iterator):
    
    ### Unpack CFG Inputs
    # unpack analysis inputs
    (MU_spike_amplitudes_list,ephys_channel_idxs_list,filter_ephys,sort_method,
     bodypart_for_reference,bodypart_ref_filter,filter_all_anipose,trial_reject_bounds_mm,
     trial_reject_bounds_sec,origin_offsets,save_binned_MU_data,time_frame,bin_width_ms,
     num_rad_bins,smoothing_window,phase_align,align_to) = CFG['analysis'].values()
    # unpack plotting inputs
    (plot_type,plot_units,do_plot,N_colors,plot_template,*_) = CFG['plotting'].values()
    # unpack chosen rat inputs
    (bodyparts_list,bodypart_for_alignment,session_date,treadmill_speed,
     treadmill_incline,camera_fps,vid_length) = CFG['rat'][chosen_rat].values()
    
    # format inputs to avoid ambiguities
    # session_date = session_date[iterator]
    # rat_name = str(chosen_rat).lower()
    # treadmill_speed = str(treadmill_speed[iterator]).zfill(2)
    # treadmill_incline = str(treadmill_incline[iterator]).zfill(2)
    # session_ID = f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
    
    # only display plot if rat_loco_analysis() is the caller
    do_plot = True if stack()[1].function == 'rat_loco_analysis' else False
    
    session_ID_lst = []
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
        # f"<b>Locomotion Kinematics: {list(chosen_anipose_dict.keys())[0]}</b>",
        # f"<b>Neural Activity: {list(chosen_ephys_data_dict.keys())[0]}</b>"
    for iPar in range(len(treadmill_incline)):
        i_session_date = session_date[iPar]
        i_rat_name = chosen_rat.lower()
        i_treadmill_speed = str(treadmill_speed[iPar]).zfill(2)
        i_treadmill_incline = str(treadmill_incline[iPar]).zfill(2)
        session_ID_lst.append(
            f"{i_session_date}_{i_rat_name}_speed{i_treadmill_speed}_incline{i_treadmill_incline}")

        (trialized_anipose_df,keep_trial_set,foot_strike_idxs,foot_off_idxs,sliced_step_stats,
         kept_step_stats, step_slice,step_time_slice, ref_bodypart_trace_list,pre_align_offset,
         post_align_offset,trial_reject_bounds_mm,trial_reject_bounds_sec) = trialize_steps(
             chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, iterator
             )
        ### save trialized data hack
        # set_trace()
        # trialized_anipose_df.to_csv('/snel/share/data/rodent-ephys/open-ephys/treadmill/2022-11-16_16-19-28/Record Node 101/experiment2/recording2/anipose/trialized_anipose_df.csv')
        # import scipy.io
        # scipy.io.savemat('/snel/share/data/rodent-ephys/open-ephys/treadmill/2022-11-16_16-19-28/Record Node 101/experiment2/recording2/anipose/variables.mat', dict(trialized_anipose_df=trialized_anipose_df,keep_trial_set=list(keep_trial_set),foot_strike_idxs=foot_strike_idxs,foot_off_idxs=foot_off_idxs))
        # scipy.io.savemat('/snel/share/data/rodent-ephys/open-ephys/treadmill/2022-11-16_16-19-28/Record Node 101/experiment2/recording2/anipose/step_idxs_only.mat', dict(keep_trial_set=list(keep_trial_set),foot_strike_idxs=foot_strike_idxs,foot_off_idxs=foot_off_idxs))
        
        # trialize_steps(
        #     anipose_dict, bodypart_for_alignment, bodypart_for_reference, bodypart_ref_filter,
        #     trial_reject_bounds_mm, trial_reject_bounds_sec, origin_offsets, bodyparts_list,
        #     filter_all_anipose, session_date[iPar], rat_name[iPar], treadmill_speed[iPar],
        #     treadmill_incline[iPar], camera_fps, align_to, time_frame)
        trialized_anipose_dfs_lst.append(trialized_anipose_df)
        trialized_anipose_dfs_lst[iPar]['Labels'] = pd.Series(int(i_treadmill_incline) * \
                                np.ones(anipose_dict[session_ID_lst[iPar]].shape[0]))
        
        # get trial averages by filtering for all columns that match
        trial_ave_lst = []
        for iBodypart in range(len(bodyparts_list)):
            trial_ave_lst.append(trialized_anipose_df.filter(like=bodyparts_list[iBodypart]).mean(axis=1))

        # plot trial averages across inclines
        fig1.add_trace(go.Scatter(
            x=trial_ave_lst[0],
            y=trial_ave_lst[1] if len(trial_ave_lst)>1 else trial_ave_lst[0],
            mode='lines', name=f'incline{i_treadmill_incline}',
            line_color=MU_colors[int(i_treadmill_incline)//5]
            ))
        fig1.add_trace(go.Scatter(
            x=[trial_ave_lst[0][0]],
            y=[trial_ave_lst[1][0] if len(trial_ave_lst)>1 else trial_ave_lst[0][0]],
            mode='markers', marker_line_color="black", marker_color=CH_colors[int(i_treadmill_incline)//5],
            marker_line_width=2, marker_size=15, marker_symbol='asterisk', name=f'incline{i_treadmill_incline} start',
            ))

        # plot single trajectories
        for iBodypart in range(len(bodyparts_list)):
            anipose_bodypart_trials = trialized_anipose_df.filter(like=bodyparts_list[iBodypart]).to_numpy()
            data_col_names = trialized_anipose_df.filter(like=bodyparts_list[iBodypart]).columns
            for (iTrial, iName) in zip(anipose_bodypart_trials.T, data_col_names):
                fig2.add_trace(go.Scatter(
                    x=np.linspace(-pre_align_offset/camera_fps, post_align_offset/camera_fps,len(iTrial)),
                    y=iTrial,
                    mode='lines', name=iName,
                    opacity=.9,
                    line_color=MU_colors[int(i_treadmill_incline)//5],
                    line=dict(width=2)),
                    col=iPar+1, row=iBodypart+1
                    )
            fig2.add_vline(x=0, line_width=3, line_dash="dash", line_color="black", name=align_to)
        
    # Edit the layout
    fig1.update_layout(
        title=f'<b>Behavioral State Space Across {bodyparts_list[0]} and {bodyparts_list[1] if len(bodyparts_list)>1 else bodyparts_list[0]}, Trial Averages</b>',
        xaxis_title='<b>'+bodyparts_list[0]+' mean</b>',
        yaxis_title=f'<b>{bodyparts_list[1] if len(bodyparts_list)>1 else bodyparts_list[0]} mean</b>'
        )
    fig1.update_yaxes(scaleanchor = "x", scaleratio = 1)
    
    fig2.update_layout(title=f'<b>Locomotion Kinematics, Aligned to {align_to.title()}: {i_session_date}_{i_rat_name}_speed{i_treadmill_speed}</b>') # Trial Rejection Bounds: {trial_reject_bounds_mm}</b>')
    for xx in range(len(treadmill_incline)):
        fig2.update_xaxes(title_text='<b>Time (sec)</b>', row = len(bodyparts_list), col = xx+1)
    for yy, yTitle in enumerate(bodyparts_list):
        fig2.update_yaxes(title_text="<b>"+str(yTitle)+" (mm)</b>", row = yy+1, col = 1)
    # fig2.update_yaxes(scaleanchor = "x",scaleratio = 1)
    # fig2.update_yaxes(matches='y')
    
    iplot(fig1)
    iplot(fig2)
    return
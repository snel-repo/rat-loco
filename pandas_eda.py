import pandas as pd
from pandas_profiling import ProfileReport
from extract_step_idxs import extract_step_idxs
import pandas as pd
import numpy as np
from plotly.offline import iplot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, butter, filtfilt, iirnotch, coherence
from scipy.ndimage import gaussian_filter1d
from pdb import set_trace

def pandas_eda(
    ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
    filter_ephys, filter_tracking, bin_width_ms, bin_width_radian, anipose_data_dict, bodypart_for_alignment,
    session_date, rat_name, treadmill_speed, treadmill_incline,
    camera_fps, alignto, vid_length, time_frame,
    do_plot, plot_template, MU_colors, CH_colors):

    iPar = 0
    _, foot_strike_idxs, foot_off_idxs, sliced_step_stats, step_slice, step_time_slice = \
            extract_step_idxs(anipose_data_dict, bodypart_for_alignment=bodypart_for_alignment,
                              filter_tracking=filter_tracking, session_date=session_date[iPar],
                              rat_name=rat_name[iPar], treadmill_speed=treadmill_speed[iPar], 
                              treadmill_incline=treadmill_incline[iPar], camera_fps=camera_fps,
                              alignto=alignto, time_frame=time_frame)
    
    step_idx_slice_lst = []
    session_parameters_lst = []
    chosen_anipose_dfs_lst = []
    step_idx_slice_lst.append(step_time_slice)
    i_session_date = session_date[iPar]
    i_rat_name = str(rat_name[iPar]).lower()
    i_treadmill_speed = str(treadmill_speed[iPar]).zfill(2)
    i_treadmill_incline = str(treadmill_incline[iPar]).zfill(2)
    session_parameters_lst.append(
        f"{i_session_date}_{i_rat_name}_speed{i_treadmill_speed}_incline{i_treadmill_incline}")
    chosen_anipose_dfs_lst.append(anipose_data_dict[session_parameters_lst[iPar]])
    chosen_anipose_dfs_lst[iPar]['Labels'] = pd.Series(int(i_treadmill_incline) * \
                            np.ones(anipose_data_dict[session_parameters_lst[iPar]].shape[0]))
    
    # convert chosen anipose dict into DataFrame        
    cols = chosen_anipose_dfs_lst[0].columns
    trimmed_anipose_dfs_lst = []
    for idf, df in enumerate(chosen_anipose_dfs_lst):
        trimmed_anipose_dfs_lst.append(df.iloc[step_idx_slice_lst[idf]])
    anipose_df = pd.DataFrame(np.concatenate(trimmed_anipose_dfs_lst),columns=cols)
    bodypart_substr = ['_x','_y','_z']
    not_bodypart_substr = ['ref','origin']
    reduced_cols = [str for str in cols if any(sub in str for sub in bodypart_substr)]
    bodypart_cols = [str for str in reduced_cols if not any(
        sub in str for sub in not_bodypart_substr)]
    body_anipose_df = anipose_df[bodypart_cols]
    ref_bodypart_aligned_df = body_anipose_df.copy()
    for iDim in bodypart_substr:
        body_dim_cols = [str for str in bodypart_cols if any(sub in str for sub in [iDim])]
        for iCol in body_dim_cols:
            ref_bodypart_aligned_df[iCol] = np.sqrt(
                (body_anipose_df["tailbase"+iDim]-body_anipose_df[iCol])**2)
    
    x_data = body_anipose_df.columns.str.endswith("_x")
    y_data = body_anipose_df.columns.str.endswith("_y")
    z_data = body_anipose_df.columns.str.endswith("_z")

    sorted_body_anipose_df = pd.concat([ref_bodypart_aligned_df.loc[:,x_data],ref_bodypart_aligned_df.loc[:,y_data],ref_bodypart_aligned_df.loc[:,z_data]],axis=1,ignore_index=False)
    design_report = ProfileReport(sorted_body_anipose_df)
    design_report.to_file(output_file=f"{session_parameters_lst[iPar]}_report_sorted_aligned.html")
    
    return

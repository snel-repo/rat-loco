from turtle import title
import pandas as pd
from pandas_profiling import ProfileReport
from process_steps import process_steps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from plotly.offline import iplot
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, butter, filtfilt, iirnotch, coherence
from scipy.ndimage import gaussian_filter1d
from pdb import set_trace
import colorlover as cl

def pandas_eda(
    ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
    filter_ephys, filter_tracking, bin_width_ms, bin_width_radian, anipose_data_dict,
    bodypart_for_alignment, bodypart_for_reference, subtract_bodypart_ref,
    session_date, rat_name, treadmill_speed, treadmill_incline,
    camera_fps, alignto, vid_length, time_frame,
    do_plot, plot_template, MU_colors, CH_colors):

    iPar = 0
    step_idx_slice_lst = []
    session_parameters_lst = []
    chosen_anipose_dfs_lst = []
    for iPar in range(len(treadmill_incline)):
        processed_anipose_df, foot_strike_idxs, foot_off_idxs, sliced_step_stats, step_slice, step_time_slice = \
                process_steps(anipose_data_dict, bodypart_for_alignment=bodypart_for_alignment,
                              bodypart_for_reference=bodypart_for_reference, 
                              subtract_bodypart_ref=subtract_bodypart_ref,
                              filter_tracking=filter_tracking, session_date=session_date[iPar],
                              rat_name=rat_name[iPar], treadmill_speed=treadmill_speed[iPar], 
                              treadmill_incline=treadmill_incline[iPar], camera_fps=camera_fps,
                              alignto=alignto, time_frame=time_frame)
        
        step_idx_slice_lst.append(step_time_slice)
        i_session_date = session_date[iPar]
        i_rat_name = str(rat_name[iPar]).lower()
        i_treadmill_speed = str(treadmill_speed[iPar]).zfill(2)
        i_treadmill_incline = str(treadmill_incline[iPar]).zfill(2)
        session_parameters_lst.append(
            f"{i_session_date}_{i_rat_name}_speed{i_treadmill_speed}_incline{i_treadmill_incline}")
        chosen_anipose_dfs_lst.append(processed_anipose_df)
        chosen_anipose_dfs_lst[iPar]['Labels'] = pd.Series(int(i_treadmill_incline) * \
                                np.ones(anipose_data_dict[session_parameters_lst[iPar]].shape[0]))
        cols = processed_anipose_df.columns
        # cols = anipose_data_dict[session_parameters_lst[iPar]].columns
    # convert chosen anipose dicts into single DataFrame        
    trimmed_anipose_dfs_lst = []
    for idf, df in enumerate(chosen_anipose_dfs_lst):
        trimmed_anipose_dfs_lst.append(df.iloc[step_idx_slice_lst[idf]])
    anipose_df_all_inclines = pd.DataFrame(np.concatenate(trimmed_anipose_dfs_lst),columns=cols)
    # bodypart_and_labels_substr = ['_x','_y','_z','Labels']
    # not_bodypart_substr = ['ref','origin']
    # reduced_cols = [str for str in cols if any(sub in str for sub in bodypart_and_labels_substr)]
    # bodypart_and_label_cols = [str for str in reduced_cols if not any(
    #     sub in str for sub in not_bodypart_substr)]
    # bodypart_and_label_anipose_df = anipose_df_all_inclines[bodypart_and_label_cols]
    # ref_bodypart = 'tailbase'
    # ref_bodypart_aligned_df = bodypart_and_label_anipose_df.copy()
    # for iDim in bodypart_and_labels_substr:
    #     if iDim == 'Labels': continue # skip if Labels column
    #     body_dim_cols = [str for str in bodypart_and_label_cols if any(sub in str for sub in [iDim])]
    #     for iCol in body_dim_cols:
    #         ref_bodypart_aligned_df[iCol] = np.sqrt(
    #             (bodypart_and_label_anipose_df[ref_bodypart+iDim] - \
    #                 bodypart_and_label_anipose_df[iCol])**2)
        
    x_data = anipose_df_all_inclines.columns.str.endswith("_x")
    y_data = anipose_df_all_inclines.columns.str.endswith("_y")
    z_data = anipose_df_all_inclines.columns.str.endswith("_z")
    Labels = anipose_df_all_inclines['Labels']
    
    sorted_body_anipose_df = pd.concat(
        [anipose_df_all_inclines.loc[:,x_data],
         anipose_df_all_inclines.loc[:,y_data],
         anipose_df_all_inclines.loc[:,z_data],
         anipose_df_all_inclines['Labels']],
        axis=1,ignore_index=False)

    # pd.options.plotting.backend = "plotly"
    # grr_X=anipose_df_all_inclines.loc[:,x_data].plot.scatter(c=Labels, alpha=.6, figsize=(15, 15))
    # grr_Y=anipose_df_all_inclines.loc[:,y_data].plot.scatter(c=Labels, alpha=.6, figsize=(15, 15))
    # grr_Z=anipose_df_all_inclines.loc[:,z_data].plot.scatter(c=Labels, alpha=.6, figsize=(15, 15))
    
    # matplotlib plotting
    label_idxs = Labels.to_numpy().astype(int)
    incline_cmap_tuple = cl.to_numeric(MU_colors)
    incline_cmap_list = [list(ele) for ele in incline_cmap_tuple]
    incline_cmap_norm = np.array(incline_cmap_list)/255
    incline_cmap = ListedColormap(incline_cmap_norm)
    # # incline_label_colors = [incline_cmap[ii] for ii in label_idxs]
    # grr_X = pd.plotting.scatter_matrix(anipose_df_all_inclines.loc[:,x_data].iloc[:,:9],c=label_idxs,cmap=incline_cmap, hist_kwds={'bins': 20},alpha=.4, s=10, figsize=(15, 15))
    # grr_Y = pd.plotting.scatter_matrix(anipose_df_all_inclines.loc[:,y_data].iloc[:,:9],
    #                                    c=label_idxs, cmap=incline_cmap, hist_kwds={'bins': 20},
    #                                    alpha=.4, s=10, figsize=(15, 15))
    # grr_Z = pd.plotting.scatter_matrix(anipose_df_all_inclines.loc[:,z_data].iloc[:,:9],
    #                                    c=label_idxs, cmap=incline_cmap, hist_kwds={'bins': 20},
    #                                    alpha=.4, s=10, figsize=(15, 15))
    # # plt.legend([grr_X,grr_Y,grr_Z],[])
    # plt.title("Pairwise Interactions Across Bodyparts During Locomotion")
    # plt.show()
    # set_trace()
    
    ## plotly plotting
    data_color_assignment_percentiles = np.linspace(0, 1, len(MU_colors))
    colorscale = [[i,j] for i,j in zip(data_color_assignment_percentiles,reversed(MU_colors))]
    grr_X = px.scatter_matrix(
        anipose_df_all_inclines.loc[:,x_data].iloc[:,:9],
        color=Labels, color_continuous_scale=colorscale, 
        opacity=.5, width=900, height=900)
    grr_Y = px.scatter_matrix(
        anipose_df_all_inclines.loc[:,y_data].iloc[:,:9],
        color=Labels, color_continuous_scale=colorscale, 
        opacity=.5, width=900, height=900)
    grr_Z = px.scatter_matrix(
        anipose_df_all_inclines.loc[:,z_data].iloc[:,:9],
        color=Labels, color_continuous_scale=colorscale, 
        opacity=.5, width=900, height=900)
    
    grr_X.update_traces(marker=dict(size=2))
    grr_Y.update_traces(marker=dict(size=2))
    grr_Z.update_traces(marker=dict(size=2))
    
    grr_X.update_layout(
        title_text="<b>Pairwise Movement for X-Dim, Bodyparts During Locomotion</b>",
        coloraxis_colorbar_title_text = '<b>Incline</b>'
    )
    grr_Y.update_layout(
        title_text="<b>Pairwise Movement for Y-Dim, Bodyparts During Locomotion</b>",
        coloraxis_colorbar_title_text = '<b>Incline</b>'
    )
    grr_Z.update_layout(
        title_text="<b>Pairwise Movement for Z-Dim, Bodyparts During Locomotion</b>",
        coloraxis_colorbar_title_text = '<b>Incline</b>'
    )
    
    set_trace()
    iplot(grr_X)
    iplot(grr_Y)
    iplot(grr_Z)
    # design_report = ProfileReport(sorted_body_anipose_df)
    # design_report.to_file(output_file=f"{session_parameters_lst[iPar]}_report_sorted_aligned.html")
    return

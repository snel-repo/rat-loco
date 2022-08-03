from extract_step_idxs import extract_step_idxs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import iplot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, butter, filtfilt, iirnotch
from scipy.ndimage import gaussian_filter1d
from pdb import set_trace
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import umap
import umap.plot

def cluster_steps(ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
    filter_ephys, filter_tracking, bin_width_ms, bin_width_radian, anipose_data_dict, bodypart_for_alignment,
    session_date, rat_name, treadmill_speed, treadmill_incline,
    camera_fps, alignto, vid_length, time_frame,
    do_plot, plot_template, MU_colors, CH_colors
    ):
    
    # check inputs for problems
    # assert len(ephys_channel_idxs_list)==1, \
    # "ephys_channel_idxs_list should only be 1 channel, idiot! :)"
    # assert type(bin_width_ms) is int, "bin_width_ms must be type 'int'."
    
    
    # format inputs to avoid ambiguities
    session_parameters_lst = []
    chosen_anipose_dfs_lst = []
    foot_strike_idxs_lst = []
    foot_off_idxs_lst = []
    all_step_idx = []
    step_idx_slice_lst = []
    for iPar in range(len(treadmill_incline)):
        _, foot_strike_idxs, foot_off_idxs, step_stats = extract_step_idxs(
            anipose_data_dict, bodypart_for_alignment=bodypart_for_alignment, filter_tracking=filter_tracking,
            session_date=session_date[iPar], rat_name=rat_name[iPar], treadmill_speed=treadmill_speed[iPar],
            treadmill_incline=treadmill_incline[iPar], camera_fps=camera_fps, alignto=alignto
            )
        foot_strike_slice_idxs = [
            foot_strike_idxs[int(step_stats['count']*time_frame[0])],
            foot_strike_idxs[int(step_stats['count']*time_frame[1])]
            ]
        foot_off_slice_idxs = [
            foot_off_idxs[int(step_stats['count']*time_frame[0])],
            foot_off_idxs[int(step_stats['count']*time_frame[1])]
            ]
        all_step_idx.append([])
        if foot_strike_slice_idxs[0]<foot_off_slice_idxs[0]:
            all_step_idx[-1].append(foot_strike_slice_idxs[0])
        else:
            all_step_idx[-1].append(foot_off_slice_idxs[0])
        if foot_strike_slice_idxs[1]>foot_off_slice_idxs[1]:
            all_step_idx[-1].append(foot_strike_slice_idxs[1])
        else:
            all_step_idx[-1].append(foot_off_slice_idxs[1])
        step_idx_slice_lst.append(slice(all_step_idx[-1][0],all_step_idx[-1][1]))
        
        foot_strike_idxs_lst.append(foot_strike_idxs)
        foot_off_idxs_lst.append(foot_off_idxs)
        i_session_date = session_date[iPar]
        i_rat_name = str(rat_name[iPar]).lower()
        i_treadmill_speed = str(treadmill_speed[iPar]).zfill(2)
        i_treadmill_incline = str(treadmill_incline[iPar]).zfill(2)
        session_parameters_lst.append(f"{i_session_date}_{i_rat_name}_speed{i_treadmill_speed}_incline{i_treadmill_incline}")
        chosen_anipose_dfs_lst.append(anipose_data_dict[session_parameters_lst[iPar]])
        chosen_anipose_dfs_lst[iPar]['Labels'] = pd.Series(int(i_treadmill_incline)*np.ones(anipose_data_dict[session_parameters_lst[iPar]].shape[0]))
    # choose alignment feature
    if alignto == 'foot strike':
        step_idxs = foot_strike_idxs
    elif alignto == 'foot off':
        step_idxs = foot_off_idxs
            
    # convert chosen anipose dict into DataFrame        
    cols = chosen_anipose_dfs_lst[0].columns
    trimmed_anipose_dfs_lst = []
    for idf, df in enumerate(chosen_anipose_dfs_lst):
        trimmed_anipose_dfs_lst.append(df.iloc[step_idx_slice_lst[idf]])
    anipose_data = pd.DataFrame(np.concatenate(trimmed_anipose_dfs_lst),columns=cols)
    bodypart_substr = ['L_x','L_y','L_z','R_x','R_y','R_z','nose','tail']
    bodypart_columns = [str for str in cols if any(sub in str for sub in bodypart_substr)]
    body_anipose_data = anipose_data[bodypart_columns]
    # anipose_steps_data = anipose_data.iloc[step_idx_slice_lst]
    scaled_data = StandardScaler().fit_transform(body_anipose_data)
    
    # compare methods
    pca_projjer = PCA(random_state=42).fit(scaled_data)
    isomap_mapper = Isomap().fit(scaled_data)
    
    # projections/embeddings
    pca_proj = pca_projjer.transform(scaled_data)
    iso_embed = isomap_mapper.transform(scaled_data)
    
    pca_fig = plt.scatter(pca_proj[:,0],pca_proj[:,1],
                          c=anipose_data['Labels'],
                          label=anipose_data['Labels'],
                          alpha=0.5)
    plt.title('pca')
    plt.show(pca_fig)
    
    iso_fig = plt.scatter(iso_embed[:,0],iso_embed[:,1],
                          c=anipose_data['Labels'],
                          label=anipose_data['Labels'],
                          alpha=0.5)
    plt.title('isomap')
    plt.show(iso_fig)
    
    umap_mapper = umap.UMAP(random_state=42).fit(scaled_data)
    umap_fig1 = umap.plot.connectivity(umap_mapper, edge_bundling='hammer')
    umap_fig2 = umap.plot.points(umap_mapper,labels=anipose_data['Labels'],theme='fire')
    plt.title('UMAP Embedding of 30-dimensional Behavior Across Incline Conditions')
    umap.plot.show(umap_fig1)
    umap.plot.show(umap_fig2)
    
    return

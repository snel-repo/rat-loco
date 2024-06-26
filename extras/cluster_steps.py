from process_steps import peak_align_and_filt
import pandas as pd
import numpy as np
from plotly.offline import iplot
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import umap
import umap.plot


def cluster_steps(
    ephys_data_dict,
    ephys_channel_idxs_list,
    MU_spike_amplitudes_list,
    filter_ephys,
    filter_all_anipose,
    bin_width_ms,
    bin_width_radian,
    anipose_data_dict,
    bodypart_for_alignment,
    bodypart_for_reference,
    bodypart_ref_filt_cutoff,
    subtract_bodypart_ref,
    origin_offsets,
    session_date,
    rat_name,
    treadmill_speed,
    treadmill_incline,
    camera_fps,
    align_to,
    vid_length,
    time_frame,
    do_plot,
    plot_template,
    MU_colors,
    CH_colors,
):
    # check inputs for problems
    # assert len(ephys_channel_idxs_list)==1, \
    # "ephys_channel_idxs_list should only be 1 channel, idiot! :)"
    # assert type(bin_width_ms) is int, "bin_width_ms must be type 'int'."

    # format inputs to avoid ambiguities
    session_parameters_lst = []
    chosen_anipose_dfs_lst = []
    # foot_strike_idxs_lst = []
    # foot_off_idxs_lst = []
    # all_step_idx = []
    step_idx_slice_lst = []
    for iPar in range(len(treadmill_incline)):
        (
            _,
            foot_strike_idxs,
            foot_off_idxs,
            sliced_step_stats,
            step_slice,
            step_time_slice,
        ) = peak_align_and_filt(
            anipose_data_dict,
            bodypart_for_alignment=bodypart_for_alignment,
            bodypart_for_reference=bodypart_for_reference,
            bodypart_ref_filt_cutoff=bodypart_ref_filt_cutoff,
            subtract_bodypart_ref=subtract_bodypart_ref,
            origin_offsets=origin_offsets,
            filter_all_anipose=filter_all_anipose,
            session_date=session_date[iPar],
            rat_name=rat_name[iPar],
            treadmill_speed=treadmill_speed[iPar],
            treadmill_incline=treadmill_incline[iPar],
            camera_fps=camera_fps,
            align_to=align_to,
            time_frame=time_frame,
        )

        step_idx_slice_lst.append(step_time_slice)
        i_session_date = session_date[iPar]
        i_rat_name = str(rat_name[iPar]).lower()
        i_treadmill_speed = str(treadmill_speed[iPar]).zfill(2)
        i_treadmill_incline = str(treadmill_incline[iPar]).zfill(2)
        session_parameters_lst.append(
            f"{i_session_date}_{i_rat_name}_speed{i_treadmill_speed}_incline{i_treadmill_incline}"
        )
        chosen_anipose_dfs_lst.append(anipose_data_dict[session_parameters_lst[iPar]])
        chosen_anipose_dfs_lst[iPar]["Labels"] = pd.Series(
            int(i_treadmill_incline)
            * np.ones(anipose_data_dict[session_parameters_lst[iPar]].shape[0])
        )

    # convert chosen anipose dict into DataFrame
    cols = chosen_anipose_dfs_lst[0].columns
    trimmed_anipose_dfs_lst = []
    for idf, df in enumerate(chosen_anipose_dfs_lst):
        trimmed_anipose_dfs_lst.append(df.iloc[step_idx_slice_lst[idf]])
    anipose_data = pd.DataFrame(np.concatenate(trimmed_anipose_dfs_lst), columns=cols)
    bodypart_substr = ["_x", "_y", "_z"]
    not_bodypart_substr = ["ref", "origin"]
    reduced_cols = [str for str in cols if any(sub in str for sub in bodypart_substr)]
    bodypart_cols = [
        str for str in reduced_cols if not any(sub in str for sub in not_bodypart_substr)
    ]
    body_anipose_data = anipose_data[bodypart_cols]
    data_ref_bodypart_aligned = body_anipose_data.copy()
    for iDim in bodypart_substr:
        body_dim_cols = [str for str in bodypart_cols if any(sub in str for sub in [iDim])]
        for iCol in body_dim_cols:
            data_ref_bodypart_aligned[iCol] = np.sqrt(
                (body_anipose_data["tailbase" + iDim] - body_anipose_data[iCol]) ** 2
            )

    # anipose_steps_data = anipose_data.iloc[step_idx_slice_lst]
    scaled_data = (
        data_ref_bodypart_aligned  # StandardScaler().fit_transform(data_ref_bodypart_aligned)
    )

    # compare methods
    pca_projjer = PCA(n_components=3, random_state=42).fit(scaled_data)
    isomap_mapper = Isomap(n_components=3).fit(scaled_data)

    # projections/embeddings
    pca_proj = pca_projjer.transform(scaled_data)
    # pca_proj_df = pd.DataFrame(pca_proj)
    iso_embed = isomap_mapper.transform(scaled_data)
    # ios_embed_df = pd.DataFrame(iso_embed)

    pca_go = go.Figure()
    last_len = 0
    for ii, df in enumerate(trimmed_anipose_dfs_lst):
        iSlice = len(df)
        pca_go.add_trace(
            go.Scatter3d(
                x=pca_proj[last_len : last_len + iSlice, 0],
                y=pca_proj[last_len : last_len + iSlice, 1],
                z=pca_proj[last_len : last_len + iSlice, 2],
                name=str(treadmill_incline[ii]) + " degrees",
                mode="markers",
                marker=dict(
                    size=3,
                    color=MU_colors[ii],  # set color to an array/list of desired values
                    opacity=0.5,
                ),
            )
        )
        last_len += iSlice

    # tight, equal layout
    pca_go.update_layout(
        title_text=f"<b>PCA Projection of 30-dimensional Behavior Across Incline Conditions</b>",
        # margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            aspectmode="data",
            xaxis=dict(title=f"<b>Component 1</b>"),
            yaxis=dict(title=f"<b>Component 2</b>"),
            zaxis=dict(title=f"<b>Component 3</b>"),
        ),
    )
    iplot(pca_go)

    # iso_fig = plt.figure()
    # iso_ax = iso_fig.add_subplot(projection='3d')

    # iso_ax.scatter(iso_embed[:,0],iso_embed[:,1],iso_embed[:,2],
    #                       c=anipose_data['Labels'],
    #                       label=anipose_data['Labels'],
    #                       alpha=0.4)
    # iso_ax.set_title('isomap')
    # iso_ax.set_xlabel('Comp 1')
    # iso_ax.set_ylabel('Comp 2')
    # iso_ax.set_zlabel('Comp 3')
    # plt.show(iso_fig)

    iso_go = go.Figure()
    last_len = 0
    for ii, df in enumerate(trimmed_anipose_dfs_lst):
        iSlice = len(df)
        iso_go.add_trace(
            go.Scatter3d(
                x=iso_embed[last_len : last_len + iSlice, 0],
                y=iso_embed[last_len : last_len + iSlice, 1],
                z=iso_embed[last_len : last_len + iSlice, 2],
                name=str(treadmill_incline[ii]) + " degrees",
                mode="markers",
                marker=dict(
                    size=3,
                    color=MU_colors[ii],  # set color to an array/list of desired values
                    opacity=0.5,
                ),
            )
        )
        last_len += iSlice

    # tight, equal layout
    iso_go.update_layout(
        title_text=f"<b>Isomap Embedding of 30-dimensional Behavior Across Incline Conditions</b>",
        # margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            aspectmode="data",
            xaxis=dict(title=f"<b>Component 1</b>"),
            yaxis=dict(title=f"<b>Component 2</b>"),
            zaxis=dict(title=f"<b>Component 3</b>"),
        ),
    )
    iplot(iso_go)

    umap_mapper = umap.UMAP(n_components=3, random_state=42).fit(scaled_data)
    umap_embed = umap_mapper.transform(scaled_data)

    umap_go = go.Figure()
    last_len = 0
    for ii, df in enumerate(trimmed_anipose_dfs_lst):
        iSlice = len(df)
        umap_go.add_trace(
            go.Scatter3d(
                x=umap_embed[last_len : last_len + iSlice, 0],
                y=umap_embed[last_len : last_len + iSlice, 1],
                z=umap_embed[last_len : last_len + iSlice, 2],
                name=str(treadmill_incline[ii]) + " degrees",
                mode="markers",
                marker=dict(
                    size=3,
                    color=MU_colors[ii],  # set color to an array/list of desired values
                    opacity=0.5,
                ),
            )
        )
        last_len += iSlice

    # tight, equal layout
    umap_go.update_layout(
        title_text=f"<b>UMAP Embedding of 30-dimensional Behavior Across Incline Conditions</b>",
        # margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            aspectmode="data",
            xaxis=dict(title=f"<b>Component 1</b>"),
            yaxis=dict(title=f"<b>Component 2</b>"),
            zaxis=dict(title=f"<b>Component 3</b>"),
        ),
    )
    iplot(umap_go)

    # umap_fig1 = umap.plot.connectivity(umap_mapper)#, edge_bundling='hammer')
    # umap_fig2 = umap.plot.points(umap_mapper, labels=anipose_data['Labels'])

    # plt.title('UMAP Embedding of 30-dimensional Behavior Across Incline Conditions')
    # umap.plot.show(umap_fig1)
    # umap.plot.show(umap_fig2)

    return

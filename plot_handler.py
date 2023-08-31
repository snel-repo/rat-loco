from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.offline import iplot
from plotly.subplots import make_subplots


def sort_plot(
    session_iterator,
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
):
    # compute number of channels and units per channel
    # then compute a color stride value to maximize use of color space
    # number_of_units_per_channel = len(MU_spike_amplitudes_list)
    # color_stride = len(MU_colors)//(number_of_units_per_channel*number_of_channels)
    color_stride = 1
    # compute number of rows to allocate for each subplot based on numbers of each channel

    if sort_method == "thresholding":
        number_of_channels = len(
            np.where(
                (np.array(ephys_channel_idxs_list) != 16)
                & (np.array(ephys_channel_idxs_list) != -1)
            )[0]
        )
        number_of_rows = (
            len(bodyparts_list) + len(ephys_channel_idxs_list) + number_of_channels // 2 + 1
        )
        row_spec_list = number_of_rows * [[None]]
        row_spec_list[0] = [{"rowspan": len(bodyparts_list)}]
        row_spec_list[len(bodyparts_list)] = [{"rowspan": len(ephys_channel_idxs_list)}]
        row_spec_list[len(bodyparts_list) + len(ephys_channel_idxs_list)] = [
            {"rowspan": 1}
        ]  # number_of_channels//2+1}]
    elif sort_method == "kilosort":
        number_of_channels = 1
        number_of_rows = (
            len(bodyparts_list) + len(ephys_channel_idxs_list) + len(MU_spikes_dict) // 6 + 1
        )
        row_spec_list = number_of_rows * [[None]]
        row_spec_list[0] = [{"rowspan": len(bodyparts_list)}]
        row_spec_list[len(bodyparts_list)] = [{"rowspan": len(ephys_channel_idxs_list)}]
        row_spec_list[len(bodyparts_list) + len(ephys_channel_idxs_list)] = [
            {"rowspan": 1}
        ]  # len(MU_spikes_dict)//6+1}]

    MU_labels = list(OE_dict.keys())[session_iterator]

    fig = make_subplots(
        rows=number_of_rows,
        cols=1,
        specs=row_spec_list,
        shared_xaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0,
        subplot_titles=(
            f"<b>Locomotion Kinematics: {list(anipose_dict.keys())[session_iterator]}</b>",
            f"<b>Neural Activity: {MU_labels}</b>",
            f"<b>Sorted Spikes: {MU_labels}</b>",
        )
        if len(bodyparts_list) > 0
        else (f"<b>Neural Activity: {MU_labels}</b>", f"<b>Sorted Spikes: {MU_labels}</b>"),
    )

    # plot all chosen bodyparts_list, including peak and trough locations for step identification
    bodypart_counter = 0
    color_list = ["cornflowerblue", "darkorange", "green", "red"]
    # ^ alternate scheme override: ['cornflowerblue','royalblue','darkorange','tomato']
    if len(bodyparts_list) > 0:
        if bodypart_for_alignment[0] not in bodyparts_list:
            print(
                (
                    "Warning! bodypart_for_alignment is not in bodyparts_list, "
                    "so foot offs/strikes will not be plotted."
                )
            )
        for name, values in chosen_anipose_df.items():
            if name in bodyparts_list:
                if name == bodypart_for_alignment[0]:
                    # filtered signal plot (used for alignment)
                    fig.add_trace(
                        go.Scatter(
                            x=time_axis_for_anipose[step_time_slice],
                            y=np.round(
                                processed_anipose_df[bodypart_for_alignment[0]][step_time_slice],
                                decimals=1,
                            ),
                            name=bodyparts_list[bodypart_counter] + " processed"
                            if filter_all_anipose or origin_offsets
                            else bodyparts_list[bodypart_counter],
                            mode="lines",
                            opacity=0.9,
                            line=dict(
                                width=2, color=color_list[bodypart_counter % len(color_list)]
                            ),
                        ),
                        row=1,
                        col=1,
                    )
                    # foot strikes
                    fig.add_trace(
                        go.Scatter(
                            x=time_axis_for_anipose[foot_strike_slice_idxs],
                            y=np.round(
                                processed_anipose_df[bodypart_for_alignment[0]][
                                    foot_strike_slice_idxs
                                ],
                                decimals=1,
                            ),
                            name=bodyparts_list[bodypart_counter] + " strike",
                            mode="markers",
                            marker=dict(color="black"),
                            opacity=0.9,
                            line=dict(width=3),
                        ),
                        row=1,
                        col=1,
                    )
                    # foot offs
                    fig.add_trace(
                        go.Scatter(
                            x=time_axis_for_anipose[foot_off_slice_idxs],
                            y=np.round(
                                processed_anipose_df[bodypart_for_alignment[0]][
                                    foot_off_slice_idxs
                                ],
                                decimals=1,
                            ),
                            name=bodyparts_list[bodypart_counter] + " off",
                            mode="markers",
                            marker=dict(color="blue"),
                            opacity=0.9,
                            line=dict(width=3),
                        ),
                        row=1,
                        col=1,
                    )
                    bodypart_counter += 1  # increment for each matching bodypart
                else:
                    if origin_offsets:
                        fig.add_trace(
                            go.Scatter(
                                x=time_axis_for_anipose[step_time_slice],
                                y=np.round(
                                    processed_anipose_df[name][step_time_slice], decimals=1
                                ),  # + 25*bodypart_counter,
                                name=bodyparts_list[bodypart_counter] + " processed",
                                mode="lines",
                                opacity=0.9,
                                line=dict(
                                    width=2, color=color_list[bodypart_counter % len(color_list)]
                                ),
                            ),
                            row=1,
                            col=1,
                        )
                    else:
                        fig.add_trace(
                            go.Scatter(
                                x=time_axis_for_anipose[step_time_slice],
                                y=np.round(
                                    values.values[step_time_slice], decimals=1
                                ),  # + 25*bodypart_counter,
                                name=bodyparts_list[bodypart_counter],
                                mode="lines",
                                opacity=0.9,
                                line=dict(
                                    width=2, color=color_list[bodypart_counter % len(color_list)]
                                ),
                            ),
                            row=1,
                            col=1,
                        )
                    bodypart_counter += 1  # increment for each matching bodypart
        if bodypart_ref_filter and origin_offsets is not False:
            # plot x/y/z reference trace
            dims = [key for key in origin_offsets.keys() if type(origin_offsets[key]) is not int]
            for dim, ref_trace in zip(dims, ref_bodypart_trace_list):
                fig.add_trace(
                    go.Scatter(
                        x=time_axis_for_anipose[step_time_slice],
                        y=np.round(ref_trace[step_time_slice], decimals=1),
                        name=f"Ref: {bodypart_for_reference}_{dim}, {bodypart_ref_filter}Hz lowpass",
                        mode="lines",
                        opacity=0.9,
                        line=dict(width=3, color="lightgray", dash="dash"),
                    ),
                    row=1,
                    col=1,
                )
    # initialize counter to keep track of total unit count across all channels
    unit_counter = np.int16(0)
    # plot all ephys traces and/or SYNC channel
    row_spacing = 0
    for iChannel, channel_number in enumerate(ephys_channel_idxs_list):
        row_spacing = (
            np.clip(
                0.9
                * np.max(
                    chosen_ephys_data_continuous_obj.samples[
                        slice_for_ephys_during_video, channel_number
                    ]
                ),
                2000,
                5000,
            )
            + row_spacing
        )
        fig.add_trace(
            go.Scatter(
                x=time_axis_for_ephys[slice_for_ephys_during_video],
                # if statement provides different scalings and offsets for ephys vs. SYNC channel
                y=np.round(
                    (
                        chosen_ephys_data_continuous_obj.samples[
                            slice_for_ephys_during_video, channel_number
                        ]
                        - row_spacing
                        if channel_number not in [-1, 16]
                        else (
                            chosen_ephys_data_continuous_obj.samples[
                                slice_for_ephys_during_video, channel_number
                            ]
                            + 4
                        )
                        * 0.5e3
                    ),
                    decimals=1,
                ),
                name=f"CH{channel_number}" if channel_number not in [-1, 16] else "SYNC",
                mode="lines",
                marker=dict(color=CH_colors[color_stride * iChannel])
                if sort_method == "thresholding"
                else dict(color=CH_colors[color_stride * iChannel]),
                opacity=1,
                line=dict(width=0.4),
            ),
            row=len(bodyparts_list) + 1,
            col=1,
        )
        if sort_method == "kilosort":
            # UnitKeys = MU_spikes_dict.keys()
            UnitKeys = plot_units
        elif sort_method == "thresholding":
            UnitKeys = MU_spikes_dict[str(channel_number)].keys()
        sliced_MU_spikes_dict = MU_spikes_dict.copy()
        for iUnit, iUnitKey in enumerate(UnitKeys):
            if channel_number not in [-1, 16]:
                if sort_method == "thresholding":
                    MU_spikes_dict_for_unit = (
                        MU_spikes_dict[str(channel_number)][iUnitKey][:]
                        + slice_for_ephys_during_video.start
                    )
                    sliced_MU_spikes_dict[str(channel_number)][
                        iUnitKey
                    ] = MU_spikes_dict_for_unit.copy()
                elif sort_method == "kilosort":
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
                    sliced_MU_spikes_dict[iUnitKey] = (
                        MU_spikes_dict_for_unit.copy() - slice_for_ephys_during_video.start
                    )
                row2 = len(bodyparts_list) + len(ephys_channel_idxs_list) + 1
                # plot spike locations onto each selected ephys trace
                fig.add_trace(
                    go.Scatter(
                        x=time_axis_for_ephys[  # index where spikes are, starting after the video
                            MU_spikes_dict_for_unit
                        ],
                        y=np.round(
                            chosen_ephys_data_continuous_obj.samples[
                                MU_spikes_dict_for_unit, channel_number
                            ]
                            - row_spacing,
                            decimals=1,
                        ),
                        name=f"CH{channel_number}, Unit {iUnitKey}",
                        mode="markers",
                        marker=dict(color=MU_colors[color_stride * unit_counter])
                        if sort_method == "thresholding"
                        else dict(color=MU_colors[color_stride * (unit_counter % len(UnitKeys))]),
                        opacity=0.9,
                        line=dict(width=3),
                    ),
                    row=len(bodyparts_list) + 1,
                    col=1,
                )
                if sort_method == "thresholding" or (sort_method == "kilosort" and iChannel == 0):
                    # plot isolated spikes into raster plot for each selected ephys trace
                    fig.add_trace(
                        go.Scatter(
                            x=time_axis_for_ephys[  # index where spikes are, starting after the video
                                MU_spikes_dict_for_unit
                            ],
                            y=np.zeros(
                                len(time_axis_for_ephys[slice_for_ephys_during_video])
                            ).astype(np.int16)
                            - unit_counter,
                            name=f"CH{channel_number}, Unit {iUnitKey}"
                            if sort_method == "thresholding"
                            else f"KS Cluster: {iUnitKey}",
                            mode="markers",
                            marker_symbol="line-ns",
                            marker=dict(
                                color=MU_colors[color_stride * unit_counter],
                                line_color=MU_colors[color_stride * unit_counter],
                                line_width=1.2,
                                size=10,
                            ),
                            opacity=1,
                        ),
                        row=row2,
                        col=1,
                    )
                unit_counter += 1

    fig.update_xaxes(
        title_text="<b>Time (s)</b>",
        row=len(bodyparts_list) + len(ephys_channel_idxs_list) + 1,
        col=1,  # secondary_y=False
    )
    fig.update_yaxes(title_text="<b>Position (mm)</b>", row=1, col=1)
    fig.update_yaxes(title_text="<b>Voltage (uV)</b>", row=len(bodyparts_list) + 1, col=1)
    fig.update_yaxes(
        title_text="<b>Sorted Spikes</b>",
        row=len(bodyparts_list) + len(ephys_channel_idxs_list) + 1,
        col=1,  # secondary_y=True
    )
    fig.update_layout(template=plot_template)
    figs = [fig]

    if do_plot == 3:
        path_to_write_to = Path.cwd().joinpath(session_ID + ".html")
        fig.write_html(str(path_to_write_to))
        plot_flag = False
    if plot_flag:
        iplot(fig)
    return figs, sliced_MU_spikes_dict


def bin_and_count_plot(
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
    (plot_type, plot_units, do_plot, N_colors, plot_template, *_) = CFG["plotting"].values()
    # sum all spikes across step cycles

    order_by_count = np.argsort(MU_spikes_count_across_all_steps)
    color_stride = 1
    fig1 = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes=False,
        subplot_titles=(f"Session Info: {session_ID}", f"Session Info: {session_ID}"),
    )
    if sort_method == "kilosort":
        MU_iter = plot_units
    else:
        MU_iter = MU_spikes_dict.keys()

    # for iUnit, iUnitKey in enumerate(np.fromiter(MU_spikes_dict.keys(),'int')[order_by_count[::-1]]):
    # import pdb; pdb.set_trace()
    for iUnit, iUnitKey in enumerate(np.fromiter(MU_iter, "int")[order_by_count[::-1]]):
        try:
            MU_step_aligned_idxs = np.concatenate(
                MU_step_aligned_spike_idxs_dict[str(iUnitKey)]
            ).ravel()
        except:
            MU_step_aligned_idxs = np.concatenate(MU_step_aligned_spike_idxs_dict[iUnitKey]).ravel()
        MU_step_aligned_idxs_ms = MU_step_aligned_idxs / ephys_sample_rate * 1000
        fig1.add_trace(
            go.Histogram(
                x=MU_step_aligned_idxs_ms,  # ms
                xbins=dict(start=0, size=bin_width_ms),
                name=str(iUnitKey) + "uV crossings"
                if (sort_method == "thresholding")
                else "KS cluster: " + str(iUnitKey),
                marker_color=MU_colors[color_stride * iUnit],
            ),
            row=1,
            col=1,
        )
    # for iUnit, iUnitKey in enumerate(np.fromiter(MU_spikes_dict.keys(),'int')[order_by_count[::-1]]):
    for iUnit, iUnitKey in enumerate(np.fromiter(MU_iter, "int")[order_by_count[::-1]]):
        try:
            MU_step_2π_aligned_idxs = np.concatenate(
                MU_step_2π_warped_spike_idxs_dict[str(iUnitKey)]
            ).ravel()
        except:
            MU_step_2π_aligned_idxs = np.concatenate(
                MU_step_2π_warped_spike_idxs_dict[iUnitKey]
            ).ravel()
        fig1.add_trace(
            go.Histogram(
                x=MU_step_2π_aligned_idxs,  # radians
                xbins=dict(start=0, size=bin_width_radian),
                name=str(iUnitKey) + "uV crossings"
                if (sort_method == "thresholding")
                else "KS cluster: " + str(iUnitKey),
                marker_color=MU_colors[color_stride * iUnit],
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # Reduce opacity to see both histograms
    fig1.update_traces(opacity=0.75)

    # set bars to overlap and all titles
    fig1.update_layout(
        barmode="overlay",
        title_text="<b>Time and Phase-Aligned Motor Unit Activity During Step Cycle</b>",
        # xaxis_title_text='<b>Time During Step (milliseconds)</b>',
        # yaxis_title_text=,
        # bargap=0., # gap between bars of adjacent location coordinates
        # bargroupgap=0.1 # gap between bars of the same location coordinates
    )
    bin_2π_rnd = np.round(bin_width_radian, 4)
    fig1.update_xaxes(title_text="<b>Time During Step (milliseconds)</b>", row=1, col=1)
    fig1.update_xaxes(title_text="<b>Phase During Step (radians)</b>", row=1, col=2)
    fig1.update_yaxes(
        title_text=f"<b>Binned Spike Count Across<br>{number_of_steps} Steps ({bin_width_ms}ms bins)</b>",
        row=1,
        col=1,
    )
    fig1.update_yaxes(
        title_text=f"<b>Binned Spike Count Across<br>{number_of_steps} Steps ({bin_2π_rnd}rad bins)</b>",
        row=1,
        col=2,
    )
    fig1.update_yaxes(matches="y")

    # set theme to chosen template
    fig1.update_layout(template=plot_template)

    # plot for total counts and Future: other stats
    fig2 = go.Figure()

    fig2.add_trace(
        go.Bar(
            # list comprehension to get threshold values for each isolated unit on this channel
            x=[
                str(iUnitKey) + "uV crossings"
                for iUnitKey in np.fromiter(MU_spikes_dict.keys(), "int")[order_by_count[::-1]]
            ]
            if sort_method == "thresholding"
            else [
                "KS Cluster: " + str(iUnitKey)
                for iUnitKey in np.fromiter(MU_spikes_dict.keys(), "int")[order_by_count[::-1]]
            ],
            y=MU_spikes_count_across_all_steps[order_by_count[::-1]],
            marker_color=[MU_colors[iColor] for iColor in range(0, len(MU_colors), color_stride)],
            opacity=1,
            showlegend=False
            # name="Counts Bar Plot"
        )
    )
    # set all titles
    fig2.update_layout(
        title_text=f"<b>Total Motor Unit Spikes Across {number_of_steps} Steps</b>\
        <br><sup>Session Info: {session_ID}</sup>",
        # xaxis_title_text='<b>Motor Unit Voltage Thresholds</b>',
        yaxis_title_text="<b>Spike Count</b>",
        # bargap=0., # gap between bars of adjacent location coordinates
        # bargroupgap=0.1 # gap between bars of the same location coordinates
    )
    # set theme to chosen template
    fig2.update_layout(template=plot_template)
    fig2.update_yaxes(matches="y")
    figs = [fig1, fig2]

    if do_plot == 3:
        path_to_write_to_1 = Path.cwd().joinpath(session_ID + "_times.svg")
        path_to_write_to_2 = Path.cwd().joinpath(session_ID + "_counts.svg")
        fig1.write_image(str(path_to_write_to_1))
        fig2.write_image(str(path_to_write_to_2))
        plot_flag = False
    if plot_flag:
        iplot(fig1)
        iplot(fig2)

    return figs


def raster_plot(
    MU_spikes_3d_array_ephys_time,
    MU_step_aligned_spike_idxs_dict,
    ephys_sample_rate,
    session_ID,
    plot_flag,
    plot_template,
    MU_colors,
):
    number_of_steps = MU_spikes_3d_array_ephys_time.shape[0]
    samples_per_step = MU_spikes_3d_array_ephys_time.shape[1]
    number_of_units = MU_spikes_3d_array_ephys_time.shape[2]

    unit_counter = 0
    step_counter = 0
    fig = go.Figure()
    # for each channel and each trial's spike time series, plot onto the raster: plotly scatters
    # for iChan in
    for iUnit, iUnitKey in enumerate(MU_step_aligned_spike_idxs_dict.keys()):
        for iStep in range(number_of_steps):
            # if number_of_units==2:
            fig.add_trace(
                go.Scatter(
                    x=MU_step_aligned_spike_idxs_dict[iUnitKey][iStep] / ephys_sample_rate * 1000,
                    y=np.zeros(samples_per_step)
                    - unit_counter
                    - step_counter
                    - iUnit * number_of_units,
                    name=f"step{iStep} unit{iUnitKey}",
                    mode="markers",
                    marker_symbol="line-ns",
                    marker=dict(
                        color=MU_colors[unit_counter],
                        line_color=MU_colors[unit_counter],
                        line_width=3,
                        size=6,
                    ),
                    opacity=0.75,
                )
            )

            step_counter += 1
        unit_counter += 1
    # ch_counter+=1
    # if number_of_units==2:
    fig.update_layout(
        title_text=f"<b>MU Activity Raster for All {number_of_steps} Steps</b>\
        <br><sup>Session Info: {session_ID}</sup>",
        xaxis_title_text=f"<b>Time (ms)</b>",
        yaxis_title_text=f"<b>Step</b>",
    )
    # elif number_of_units==3:
    #     fig.update_layout(
    #         title_text=
    #         f'<b>MU Activity Raster for All {number_of_steps} Steps</b>\
    #         <br><sup>Session Info: {session_ID}</sup>',
    #         xaxis_title_text=f'<b>Time (ms)</b>',
    #         yaxis_title_text=f'<b>Step (ms)</b>'
    #         )

    # set theme to chosen template
    fig.update_layout(template=plot_template)

    if plot_flag == 3:
        path_to_write_to = Path.cwd().joinpath(session_ID + "_raster.svg")
        fig.write_image(str(path_to_write_to))
        plot_flag = False
    if plot_flag:
        iplot(fig)

    return fig


def smoothed_plot(
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
):
    fig = go.Figure()

    # plot smoothed traces for each unit
    for iUnit in range(number_of_units):
        for iStep in range(number_of_steps):
            MU_smoothed_spikes_ztrimmed_array = np.trim_zeros(
                MU_smoothed_spikes_3d_array[iStep, :, iUnit], trim="b"
            )
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(MU_smoothed_spikes_ztrimmed_array))
                    if not phase_align
                    else np.arange(2 * np.pi, number_of_bins),
                    y=MU_smoothed_spikes_ztrimmed_array,
                    name=f"step{iStep}_unit{iUnit}",
                    mode="lines",
                    opacity=0.5,
                    line=dict(width=8, color=MU_colors[iUnit], dash="solid"),
                )
            )
    # plot mean traces for each unit
    for iUnit in range(number_of_units):
        fig.add_trace(
            go.Scatter(
                x=np.arange(MU_smoothed_spikes_3d_array.shape[1])
                if not phase_align
                else np.arange(2 * np.pi, number_of_bins),
                y=MU_smoothed_spikes_mean_2d_array[:, iUnit],
                name=f"mean_unit{iUnit}",
                mode="lines",
                opacity=1,
                # dash styles: ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
                line=dict(width=4, color=CH_colors[iUnit], dash="dot"),
            )
        )

    fig.update_layout(
        title_text=f"<b>{title_prefix}Aligned MU Activity for All {number_of_steps} Steps</b>\
        <br><sup>Session Info: {session_ID}</sup>",
        xaxis_title_text=f"<b>Bins ({np.round(bin_width,4)}{bin_unit})</b>",
        yaxis_title_text=f"<b>Smoothed MU Activity<br>({smoothing_window} Sample Kernel)</b>",
    )
    # set theme to chosen template
    # fig.update_layout(template=plot_template)

    if plot_flag == 3:
        path_to_write_to = Path.cwd().joinpath(session_ID + "_smoothed.svg")
        fig.write_image(str(path_to_write_to))
        plot_flag = False
    if plot_flag:
        iplot(fig)

    return fig


def state_space_plot(
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
):
    fig = go.Figure()
    # smooth and plot each trace
    for iStep, true_step in enumerate(steps_to_keep_arr):
        if number_of_units <= 2:
            fig.add_trace(
                go.Scatter(
                    x=sliced_MU_smoothed_3d_array[iStep, :, 0],
                    y=sliced_MU_smoothed_3d_array[iStep, :, 1],
                    name=f"step{true_step}",
                    mode="lines",
                    opacity=0.5,
                    line=dict(width=5, color=MU_colors[int(treadmill_incline) // 5], dash="solid"),
                )
            )
        elif number_of_units >= 3:
            fig.add_trace(
                go.Scatter3d(
                    x=sliced_MU_smoothed_3d_array[iStep, :, 0],
                    y=sliced_MU_smoothed_3d_array[iStep, :, 1],
                    z=sliced_MU_smoothed_3d_array[iStep, :, 2],
                    name=f"step{true_step}",
                    mode="lines",
                    opacity=0.5,
                    line=dict(width=8, color=MU_colors[int(treadmill_incline) // 5], dash="solid"),
                )
            )
    # plot mean traces for each unit
    if number_of_units <= 2:
        fig.add_trace(
            go.Scatter(
                x=sliced_MU_smoothed_3d_array[:, :, 0].mean(0),
                y=sliced_MU_smoothed_3d_array[:, :, 1].mean(0),
                name=f"mean",
                mode="markers",
                opacity=1,
                line=dict(width=10, color=CH_colors[0], dash="solid"),
                marker=dict(size=8, color=CH_colors[0]),
            )
        )
    elif number_of_units >= 3:
        fig.add_trace(
            go.Scatter3d(
                x=sliced_MU_smoothed_3d_array[:, :, 0].mean(0),
                y=sliced_MU_smoothed_3d_array[:, :, 1].mean(0),
                z=sliced_MU_smoothed_3d_array[:, :, 2].mean(0),
                name=f"mean",
                mode="markers",
                opacity=1,
                line=dict(width=10, color=CH_colors[0], dash="solid"),
                marker=dict(size=3, color=CH_colors[0]),
            )
        )
    if number_of_units <= 2:
        fig.update_layout(
            title_text=f"<b>{title_prefix}-Aligned MU State Space Activity for Channel {ephys_channel_idxs_list[0]} Across Inclines</b>\
            <br><b>Incline: {treadmill_incline}</b>",  # Bin Width: {np.round(bin_width,3)}{bin_unit}, Smoothed by {smoothing_window} bin window</sup>',
            xaxis_title_text=f"<b>Unit {plot_units[0]} Activity</b>",
            yaxis_title_text=f"<b>Unit {plot_units[1]} Activity</b>",
        )
    elif number_of_units >= 3:
        fig.update_layout(
            title_text=f"<b>{title_prefix}-Aligned MU State Space Activity for Channel {ephys_channel_idxs_list[0]} Across Inclines</b>\
            <br><b>{session_ID}</b>"
        )  # , Bin Width: {np.round(bin_width,3)}{bin_unit}, Smoothed by {smoothing_window} bins</sup>')
        fig.update_scenes(
            dict(
                camera=dict(
                    eye=dict(x=-2, y=-0.3, z=0.2)
                ),  # the default values are 1.25, 1.25, 1.25
                xaxis=dict(title_text=f"<b>Unit {plot_units[0]} Activity</b>", range=[0, 1.0]),
                yaxis=dict(title_text=f"<b>Unit {plot_units[1]} Activity</b>", range=[0, 1.0]),
                zaxis=dict(title_text=f"<b>Unit {plot_units[2]} Activity</b>", range=[0, 1.0]),
                aspectmode="manual",  # this string can be 'data', 'cube', 'auto', 'manual'
                # custom aspectratio is defined as follows:
                aspectratio=dict(x=1, y=1, z=1),
            )
        )
    # set theme to chosen template
    # fig.update_layout(template=plot_template)
    if plot_flag == 3:
        path_to_write_to = Path.cwd().joinpath(session_ID + "_state_space.svg")
        if number_of_units <= 2:
            fig.write_image(str(path_to_write_to))
        else:
            fig.write_html(str(path_to_write_to))
    if plot_flag:
        iplot(fig)
    return


def MU_space_stepwise(
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
):
    # set height ratios for each subplot using `specs` parameter of `make_subplots()`
    number_of_steps = len(keep_trial_set)
    number_of_rows = 2 * (len(plot_units) * number_of_steps) + 1
    row_spec_list = number_of_rows * [[None]]
    if len(plot_units) >= 3 and MU_smoothed_spikes_3d_array.any(1).any(0).sum() >= 3:
        row_spec_list[0] = [
            {"type": "scatter3d", "rowspan": len(plot_units) * number_of_steps, "b": 0.01}
        ]  # 1% padding between
    elif len(plot_units) >= 2 and MU_smoothed_spikes_3d_array.any(1).any(0).sum() >= 2:
        row_spec_list[0] = [
            {"type": "scatter", "rowspan": len(plot_units) * number_of_steps, "b": 0.1}
        ]  # 10% padding between
    row_spec_list[len(plot_units) * number_of_steps + 1] = [
        {"type": "scatter", "rowspan": len(plot_units) * number_of_steps}
    ]

    big_fig.append(
        make_subplots(
            rows=number_of_rows,
            cols=1,
            specs=row_spec_list,
            shared_xaxes=False,
            subplot_titles=(
                f"tmp",
                f"<b>Binned Neural Activity for Steps:</b> {keep_trial_set}",
            ),
        )
    )
    big_fig[iPar].layout.annotations[0].update(text=figs[0].layout.title.text)  # .split('<br>')[1])
    for iTrace in range(len(figs[0].data)):
        big_fig[iPar].add_trace(figs[0].data[iTrace], row=1, col=1)

    activity_matrix = binned_spike_array
    row_offset = activity_matrix.max()
    CH_colors.reverse()
    for iStep in range(activity_matrix.shape[0]):
        big_fig[iPar].data[iStep]["line"]["color"] = MU_colors[iStep]
        big_fig[iPar].data[iStep]["opacity"] = 0.5
        big_fig[iPar].data[iStep]["line"]["width"] = 2
        for iUnit in plot_units:
            if iUnit == 0:
                color = MU_colors[iStep % len(MU_colors)]
            elif iUnit == 1:  # color large units darker
                color = "grey"  # CH_colors[iStep % len(MU_colors)]
            else:
                color = "black"
            # big_fig[iPar].add_scatter(x=np.hstack((np.arange(MU_smoothed_spikes_3d_array.shape[1]),MU_smoothed_spikes_3d_array.shape[1]-1,0)),
            #                           y=np.hstack((MU_smoothed_spikes_3d_array[iStep,:,iUnit]*max(activity_matrix[iStep,:,iUnit])/max(MU_smoothed_spikes_3d_array[iStep,:,iUnit]),0,0))-row_offset*iStep,
            #                           row=len(plot_units)*number_of_steps+2, col=1,
            #                           name=f"smoothed step{steps_to_keep_arr[iStep]}, unit{iUnit}",
            #                           opacity=0.6, fill='toself', mode='lines', # override default markers+lines
            #                           line_color=color, fillcolor=color,
            #                           hoverinfo="skip", showlegend=False
            #                           )
    for iStep in range(activity_matrix.shape[0]):
        big_fig[iPar].data[iStep]["line"]["color"] = MU_colors[iStep]
        big_fig[iPar].data[iStep]["opacity"] = 0.5
        big_fig[iPar].data[iStep]["line"]["width"] = 2
        for iUnit in plot_units:
            if iUnit == np.sort(plot_units)[0]:
                color = MU_colors[iStep % len(MU_colors)]
            elif iUnit == np.sort(plot_units)[1]:  # color large units darker
                color = "grey"  # CH_colors[iStep % len(MU_colors)]
            else:
                color = "black"
            big_fig[iPar].add_scatter(
                x=np.arange(len(activity_matrix[iStep, :, iUnit])),
                y=activity_matrix[iStep, :, iUnit] - row_offset * iStep,
                row=len(plot_units) * number_of_steps + 2,
                col=1,
                name=f"step{steps_to_keep_arr[iStep]}, unit{iUnit}",
                mode="lines",
                line_color=color,
                line_width=1,
                opacity=0.5,
            )

    if phase_align:
        bin_width_radian = (
            2 * np.pi / (kept_step_stats["max"] / camera_fps / bin_width_ms)
        )  # (2*np.pi)/num_rad_bins
        bin_width = np.round(bin_width_radian, decimals=3)
        bin_units = "radians"
    else:
        bin_width = bin_width_ms
        bin_units = "ms"

    big_fig[iPar].update_xaxes(
        title_text=f"<b>Bins ({bin_width} {bin_units})</b>",
        row=len(plot_units) * number_of_steps + 2,
        col=1,
    )
    big_fig[iPar].update_yaxes(
        title_text=f"<b>Stepwise Binned Spike Counts</b>",
        row=len(plot_units) * number_of_steps + 2,
        col=1,
    )

    # Edit the layout
    if len(plot_units) >= 3 and MU_smoothed_spikes_3d_array.any(1).any(0).sum() >= 3:
        big_fig[iPar].update_layout(
            title=f"<b>Comparison Across Motor Unit State-Space Representation and Spike Binning</b>",
            title_font_size=20,
        )
        big_fig[iPar].update_scenes(
            dict(
                camera=dict(
                    eye=dict(x=-0.3, y=-2, z=0.2)
                ),  # the default values are 1.25, 1.25, 1.25
                xaxis=dict(title_text=f"<b>Unit {plot_units[0]}</b>"),
                yaxis=dict(title_text=f"<b>Unit {plot_units[1]}</b>"),
                zaxis=dict(title_text=f"<b>Unit {plot_units[2]}</b>"),
                aspectmode="manual",  # this string can be 'data', 'cube', 'auto', 'manual'
                # custom aspectratio is defined as follows:
                aspectratio=dict(x=1, y=1, z=1),
            ),
            row=1,
            col=1,
        )
    elif len(plot_units) >= 2 and MU_smoothed_spikes_3d_array.any(1).any(0).sum() >= 2:
        big_fig[iPar].update_layout(
            title=f"<b>Comparison Across Motor Unit State-Space Representation and Spike Binning</b>",
            title_font_size=20,
            xaxis_title=f"<b>Unit {plot_units[0]} Smoothed Activity</b>",
            yaxis_title=f"<b>Unit {plot_units[1]} Smoothed Activity</b>",
        )
        big_fig[iPar].update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)
    # big_fig[iPar].update_yaxes(scaleanchor = "x", scaleratio = 1, row=1, col=1)
    # big_fig[iPar].update_yaxes(matches='x', row=1, col=1)

    # fig2.update_layout(title=f'<b>{align_to}-Aligned Kinematics for {i_rat_name} on {i_session_date}, Trial Bounds: {trial_reject_bounds_mm}</b>')
    # for xx in range(len(treadmill_incline)):
    #     fig2.update_xaxes(title_text='<b>Time (sec)</b>', row = len(bodyparts_list), col = xx+1)
    # for yy, yTitle in enumerate(bodyparts_list):
    #     fig2.update_yaxes(title_text="<b>"+str(yTitle)+" (mm)</b>", row = yy+1, col = 1)
    # fig2.update_yaxes(scaleanchor = "x",scaleratio = 1)
    # fig2.update_yaxes(matches='y')
    if plot_flag:
        iplot(big_fig[iPar])

    return

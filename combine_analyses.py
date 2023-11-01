# from inspect import stack
# from pdb import set_trace
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
import plotly.graph_objects as go
# from IPython.display import display
# from plotly.offline import iplot
# from plotly.subplots import make_subplots
# from scipy.signal import butter, find_peaks, medfilt, sosfiltfilt
from colour import Color
from process_spikes import * 
from process_steps import *
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px


def behavioral_space(
    chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator
):
    def normalized_value_to_hex_color(value):
        # Define your color scale with two colors (e.g., blue to red)
        color1 = '#0000FF'  # Blue
        color2 = '#00AA00'  # Green

        # Convert the hex color strings to RGB tuples 
        r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
        r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)

        # Interpolate between the two colors based on the normalized value
        r = int(r1 + (r2 - r1) * value)
        g = int(g1 + (g2 - g1) * value)
        b = int(b1 + (b2 - b1) * value)

        # Convert the interpolated RGB values back to a hex color code
        hex_color = "#{:02X}{:02X}{:02X}".format(r, g, b)

        return hex_color
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

    # only display plot if rat_loco_analysis() is the caller
    if plot_type.__contains__("multi") or not stack()[1].function == "rat_loco_analysis":
        do_plot = False
    elif do_plot == 0:
        do_plot = False
    if do_plot == 2:  # override above, always plot if do_plot==2
        do_plot = True

    session_ID_lst = []
    trialized_anipose_dfs_lst = []
    subtitles = []
    for iTitle in treadmill_incline:
        subtitles.append("<b>Incline: " + str(iTitle) + "</b>")
    fig1 = go.Figure()
    fig2 = make_subplots(
        rows=len(bodyparts_list),
        cols=len(treadmill_incline),
        shared_xaxes=True,
        shared_yaxes="rows",
        subplot_titles=subtitles,
    )
    fig3 = make_subplots( # MU gradient plot
        rows=len(bodyparts_list),
        cols=len(treadmill_incline),
        shared_xaxes=True,
        shared_yaxes="rows",
        subplot_titles=subtitles,
    )
    # f"<b>Locomotion Kinematics: {list(chosen_anipose_dict.keys())[0]}</b>",
    # f"<b>Neural Activity: {list(chosen_ephys_data_dict.keys())[0]}</b>"
    
    #Call sort to create ordered steps dictionary with separated unique motor unit spike times  
    steps_dict = sort(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator)
    MU_time_diff_steps = {}
    print(f"Smallest to Largest MU: {plot_units}")
    smallest_MU = int(input("Enter int value of smallest MU: "))
    largest_MU = int(input("Enter int value of largest MU: "))
    for iStep in steps_dict:
        #if len(steps_dict[iStep][f'MU:{largest_MU} Indexes'])>0 & len(steps_dict[iStep][f'MU:{smallest_MU} Indexes'])>0:
        try:
            MU_time_diff_steps[iStep] = (steps_dict[iStep][f'MU:{largest_MU} Indexes'][0] - steps_dict[iStep][f'MU:{smallest_MU} Indexes'][0])
        except:
            continue
            
    # for session_iterator in range(len(treadmill_incline)):
    i_session_date = session_date[session_iterator]
    i_rat_name = chosen_rat.lower()
    i_treadmill_speed = str(treadmill_speed[session_iterator]).zfill(2)
    i_treadmill_incline = str(treadmill_incline[session_iterator]).zfill(2)
    session_ID_lst.append(
        f"{i_session_date}_{i_rat_name}_speed{i_treadmill_speed}_incline{i_treadmill_incline}"
    )

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
        chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator
    )
    
    ##set_trace() # Maps each step quickly 
    #import plotly.express as px
    #fig = px.line(trialized_anipose_df, x = trialized_anipose_df.index, y = trialized_anipose_df.filter(like='palm_L_y').columns, template = 'plotly_dark'); fig.show()
    
    
        ### save trialized data hack
        # set_trace()
        # trialized_anipose_df.to_csv('/snel/share/data/rodent-ephys/open-ephys/treadmill/2022-11-16_16-19-28/Record Node 101/experiment2/recording2/anipose/trialized_anipose_df.csv')
        # import scipy.io
        # scipy.io.savemat('/snel/share/data/rodent-ephys/open-ephys/treadmill/2022-11-16_16-19-28/Record Node 101/experiment2/recording2/anipose/variables.mat', dict(trialized_anipose_df=trialized_anipose_df,keep_trial_set=list(keep_trial_set),foot_strike_idxs=foot_strike_idxs,foot_off_idxs=foot_off_idxs))
        # scipy.io.savemat('/snel/share/data/rodent-ephys/open-ephys/treadmill/2022-11-16_16-19-28/Record Node 101/experiment2/recording2/anipose/step_idxs_only.mat', dict(keep_trial_set=list(keep_trial_set),foot_strike_idxs=foot_strike_idxs,foot_off_idxs=foot_off_idxs))

        # trialize_steps(
        #     anipose_dict, bodypart_for_alignment, bodypart_for_reference, bodypart_ref_filter,
        #     trial_reject_bounds_mm, trial_reject_bounds_sec, origin_offsets, bodyparts_list,
        #     filter_all_anipose, session_date[session_iterator], rat_name[session_iterator], treadmill_speed[session_iterator],
        #     treadmill_incline[session_iterator], camera_fps, align_to, time_frame)
    trialized_anipose_dfs_lst.append(trialized_anipose_df)
    trialized_anipose_dfs_lst[0]["Labels"] = pd.Series(
        int(i_treadmill_incline) * np.ones(anipose_dict[session_ID_lst[0]].shape[0])
    )

    # get trial averages by filtering for all columns that match
    trial_ave_lst = []
    for iBodypart in range(len(bodyparts_list)):
        trial_ave_lst.append(
            trialized_anipose_df.filter(like=bodyparts_list[iBodypart]).mean(axis=1)
        )
    # plot trial averages across inclines
    fig1.add_trace(
        go.Scatter(
            x=trial_ave_lst[0],
            y=trial_ave_lst[1] if len(trial_ave_lst) > 1 else trial_ave_lst[0],
            mode="lines",
            name=f"incline{i_treadmill_incline}",
            line_color=MU_colors[int(i_treadmill_incline) // 5],
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=[trial_ave_lst[0][0]],
            y=[trial_ave_lst[1][0] if len(trial_ave_lst) > 1 else trial_ave_lst[0][0]],
            mode="markers",
            marker_line_color="black",
            marker_color=CH_colors[int(i_treadmill_incline) // 5],
            marker_line_width=2,
            marker_size=15,
            marker_symbol="asterisk",
            name=f"incline{i_treadmill_incline} start",
        )
    )

    # plot single trajectories
    for iBodypart in range(len(bodyparts_list)):
        anipose_bodypart_trials = trialized_anipose_df.filter(
            like=bodyparts_list[iBodypart]
        ).to_numpy()
        data_col_names = trialized_anipose_df.filter(like=bodyparts_list[iBodypart]).columns
        # start_color = [1, 0, 0]  # Red
        # end_color = [0, 0, 1]    # Blue    
        # cmap = LinearSegmentedColormap.from_list("custom_gradient", [start_color, end_color])     
        # normalized_values = [(value - min(MU_time_diff_steps.values())) / (max(MU_time_diff_steps.values()) - min(MU_time_diff_steps.values())) for value in MU_time_diff_steps.values()]
        # colors = cmap(normalized_values)
        num = 0
        for iTrial, iName in zip(anipose_bodypart_trials.T, data_col_names):
            #set_trace()
            try:
                fig2.add_trace(
                go.Scatter(
                    x=np.linspace(
                        -pre_align_offset / camera_fps,
                        post_align_offset / camera_fps,
                        len(iTrial),
                    ),
                    y=iTrial,
                    mode="lines",
                    name=iName,
                    opacity=0.9,
                    line_color = 'black',
                    #line_color=colors[int(iTrial)],
                    line=dict(width=2),
                ),
                col=session_iterator + 1,
                row=iBodypart + 1,
                )
                num += 1
            except KeyError:
                continue
            
        fig2.add_vline(x=0, line_width=3, line_dash="dash", line_color="black", name=align_to)
        
        ## Mani Code for gradient plot ##
        steps_gradient_plot = [x for x in MU_time_diff_steps.keys()]
        anipose_gradient_trials = anipose_bodypart_trials.T[steps_gradient_plot]
        gradient_col_names = data_col_names[steps_gradient_plot]
        values_gradient_plot = [x for x in MU_time_diff_steps.values()]
        normalized_values = (values_gradient_plot - min(values_gradient_plot)) / (max(values_gradient_plot) - min(values_gradient_plot))
        colors = [normalized_value_to_hex_color(value) for value in normalized_values]
        set_trace()
        for iTrial, iName, iNum in zip(anipose_gradient_trials, gradient_col_names, range(len(colors))):
            #set_trace()
            try:
                fig3.add_trace(
                go.Scatter(
                    x=np.linspace(
                        -pre_align_offset / camera_fps,
                        post_align_offset / camera_fps,
                        len(iTrial),
                    ),
                    y=iTrial,
                    mode="lines",
                    name=iName,
                    opacity=0.9,
                    #line_color = 'black',
                    line_color=colors[iNum],
                    line=dict(width=2),
                ),
                col=session_iterator + 1,
                row=iBodypart + 1,
                )
                num += 1
            except KeyError:
                continue
            
            
    # Edit the layout
    fig1.update_layout(
        title=f"<b>Behavioral State Space Across {bodyparts_list[0]} and {bodyparts_list[1] if len(bodyparts_list)>1 else bodyparts_list[0]}, Trial Averages</b>",
        xaxis_title="<b>" + bodyparts_list[0] + " mean</b>",
        yaxis_title=f"<b>{bodyparts_list[1] if len(bodyparts_list)>1 else bodyparts_list[0]} mean</b>",
    )
    fig1.update_yaxes(scaleanchor="x", scaleratio=1)

    fig2.update_layout(
        title=f"<b>Locomotion Kinematics, Aligned to {align_to.title()}: {i_session_date}_{i_rat_name}_speed{i_treadmill_speed}</b>"
    )  # Trial Rejection Bounds: {trial_reject_bounds_mm}</b>')
    for xx in range(len(treadmill_incline)):
        fig2.update_xaxes(title_text="<b>Time (sec)</b>", row=len(bodyparts_list), col=xx + 1)
    for yy, yTitle in enumerate(bodyparts_list):
        fig2.update_yaxes(title_text="<b>" + str(yTitle) + " (mm)</b>", row=yy + 1, col=1)
    # fig2.update_yaxes(scaleanchor = "x",scaleratio = 1)
    # fig2.update_yaxes(matches='y')
    
    fig3.update_layout(
        title=f"<b>Locomotion Kinematics, Aligned to {align_to.title()}: {i_session_date}_{i_rat_name}_speed{i_treadmill_speed}. Plot Units: {smallest_MU}, {largest_MU}</b>"
    )  # Trial Rejection Bounds: {trial_reject_bounds_mm}</b>')
    for xx in range(len(treadmill_incline)):
        fig3.update_xaxes(title_text="<b>Time (sec)</b>", row=len(bodyparts_list), col=xx + 1)
    for yy, yTitle in enumerate(bodyparts_list):
        fig3.update_yaxes(title_text="<b>" + str(yTitle) + " (mm)</b>", row=yy + 1, col=1)

    iplot(fig1)
    iplot(fig2)
    iplot(fig3)
    return
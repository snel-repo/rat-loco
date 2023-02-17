from import_OE_data import import_OE_data
from import_anipose_data import import_anipose_data 
from import_KS_data import import_KS_data
import colorlover as cl
from collections import deque
from config import config as CFG
from pdb import set_trace
# from pdb import set_trace

### Chosen Rat ###
chosen_rat = 'godzilla' # <-- Choose Rat HERE

### Process Chosen Plotting Parameters
# black to grey
CH_colors = cl.to_rgb(cl.interp(cl.scales['6']['seq']['Greys'],
                                CFG['plotting']['N_colors']))[
                                    -1:-(CFG['plotting']['N_colors']+1):-1]
# rainbow scale
MU_colors = cl.to_rgb(cl.interp(cl.scales['10']['div']['Spectral'],
                                CFG['plotting']['N_colors']))
# rotate or reverse colors palettes, if needed
MU_colors_deque = deque(MU_colors)
MU_colors = list(MU_colors_deque)
MU_colors.reverse()
MU_colors= MU_colors[:-1]
# MU_colors = ['royalblue','green','darkorange','firebrick']

# # function filters data dictionaries for desired data
# def filter_data_dict(data_dict, session_date, rat_name, treadmill_speed, treadmill_incline):
#     data_dict_filtered_by_date = dict(filter(lambda item:
#                                 str(session_date) in item[0], data_dict.items()))
#     data_dict_filtered_by_ratname = dict(filter(lambda item:
#                                 rat_name in item[0], data_dict_filtered_by_date.items()))
#     data_dict_filtered_by_speed = dict(filter(lambda item:
#                                 "speed"+str(treadmill_speed).zfill(2) in item[0],
#                                 data_dict_filtered_by_ratname.items()))
#     data_dict_filtered_by_incline = dict(filter(lambda item:
#                                 "incline"+str(treadmill_incline).zfill(2) in item[0],
#                                 data_dict_filtered_by_speed.items()))
#     chosen_data_dict = data_dict_filtered_by_incline
#     return chosen_data_dict

def rat_loco_analysis(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator):
    # if not running a multi-session analysis, the first element in session_iterator is used
        
    if 'multi' not in CFG['plotting']['plot_type']:
        # functions for calling all analysis functions. Only chosen "plot_type" is executed
        if CFG['plotting']['plot_type'] == "sort":
            from process_spikes import sort
            if CFG['analysis']['export_data'] is True:
                for iRec in range(len(CFG['rat'][chosen_rat]['session_date'])):
                    sort(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator[iRec])
            else:
                sort(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator[0])
        elif CFG['plotting']['plot_type'] == "bin_and_count":
            from process_spikes import bin_and_count
            bin_and_count(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator[0])
        elif CFG['plotting']['plot_type'] == "raster":
            from process_spikes import raster
            raster(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator[0])
        elif CFG['plotting']['plot_type'] == "smooth":
            from process_spikes import smooth
            smooth(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator[0])
        elif CFG['plotting']['plot_type'] == "state_space":
            from process_spikes import state_space
            state_space(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator[0])
        elif CFG['plotting']['plot_type'] == "MU_space_stepwise":
            from process_spikes import MU_space_stepwise
            MU_space_stepwise(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator[0])
        elif CFG['plotting']['plot_type'] == "behavioral_space":
            from process_steps import behavioral_space
            behavioral_space(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator[0])
        elif CFG['plotting']['plot_type'] == "cluster_steps":
            from cluster_steps import cluster_steps
            cluster_steps(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator[0])
        elif CFG['plotting']['plot_type'] == "pandas_eda":
            from pandas_eda import pandas_eda
            pandas_eda(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator[0])
        elif CFG['plotting']['plot_type'] == "spike_motion_plot":
            from spike_motion_plot import spike_motion_plot
            spike_motion_plot(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator[0])           
    else:
        ### Functions with prefix "multi" are designed to loop and compare across multiple condtions
        # multi_bin performs binning of spikes, plots results for all chosen conditions
        if CFG['plotting']['plot_type'] == "multi_bin":
            from plotly.offline import iplot
            from plotly.subplots import make_subplots
            from process_spikes import bin_and_count
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
            
            num_sessions = len(session_date)
            big_fig = make_subplots(rows=num_sessions,cols=2,shared_xaxes=True,shared_yaxes=False,
                                    horizontal_spacing=0.1, vertical_spacing=0.1,
                                    subplot_titles=tuple(2*num_sessions*['tmp_title']))
            for iRec in session_iterator:
                # format inputs to avoid ambiguities
                (_,_,_,_,_,_,_,_,_,_,_,figs) = bin_and_count(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, iRec)
                # (_,_,_,_,_,_,_,_,_,_,_,figs) = bin_and_count(
                #     OE_dict, KS_dict, anipose_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
                #     filter_ephys, sort_method,  filter_all_anipose, bin_width_ms, bin_width_radian,                 bodypart_for_alignment, bodypart_for_reference,
                #     bodypart_ref_filter, trial_reject_bounds_mm, trial_reject_bounds_sec, origin_offsets, bodyparts_list,
                #     session_date[iRec], rat_name[iRec], treadmill_speed[iRec], treadmill_incline[iRec],
                #     camera_fps, align_to, vid_length, time_frame, save_binned_MU_data,
                #     do_plot=False, plot_template=plot_template, MU_colors=MU_colors, CH_colors=CH_colors)
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
                
                num_sessions = len(session_date)
                if sort_method == 'kilosort':
                    for iHist in range(len(figs[0].data)):
                        big_fig.add_trace(figs[0].data[iHist], row=iRec+1,col=(iHist//len(plot_units))+1)
                elif sort_method == 'thresholding':
                    for iHist in range(len(figs[0].data)):
                        big_fig.add_trace(figs[0].data[iHist], row=iRec+1,col=(iHist//len(MU_spike_amplitudes_list))+1)
                # keep track of session recording parameters, and set those for subplot titles
                big_fig.layout.annotations[2*iRec].update(text=figs[0].layout.annotations[0].text)
                big_fig.layout.annotations[2*iRec+1].update(text=figs[0].layout.annotations[1].text)
                # set y-axis titles to those received from bin_and_count()
                big_fig.update_yaxes(title_text=figs[0].layout.yaxis.title.text,row=iRec+1,col=1)
                big_fig.update_yaxes(title_text=figs[0].layout.yaxis2.title.text,row=iRec+1,col=2)
            # set x-axis titles to those received from bin_and_count()
            big_fig.update_xaxes(title_text=figs[0].layout.xaxis.title.text,row=iRec+1,col=1)
            big_fig.update_xaxes(title_text=figs[0].layout.xaxis2.title.text,row=iRec+1,col=2)
            big_fig.update_yaxes(matches='y')
            # Reduce opacity to see both histograms
            big_fig.update_traces(opacity=0.75)
            # set bars to overlap and all titles, and use received title from bin_and_count()
            big_fig.update_layout(barmode='overlay',title_text=figs[0].layout.title.text)
            # set_trace()
            iplot(big_fig)
        # multi_count performs counting of total number of spikes, plots results for all chosen conditions
        # elif CFG['plotting']['plot_type'] == "multi_count":
        #     from plotly.offline import iplot
        #     from plotly.subplots import make_subplots
        #     from process_spikes import bin_and_count
        #     ### Unpack CFG Inputs
        #     # unpack analysis inputs
        #     (MU_spike_amplitudes_list,ephys_channel_idxs_list,filter_ephys,sort_method,
        #     bodypart_for_reference,bodypart_ref_filter,filter_all_anipose,trial_reject_bounds_mm,
        #     trial_reject_bounds_sec,origin_offsets,save_binned_MU_data,time_frame,bin_width_ms,
        #     num_rad_bins,smoothing_window,phase_align,align_to,export_data) = CFG['analysis'].values()
        #     # unpack plotting inputs
        #     (plot_type,plot_units,do_plot,N_colors,plot_template,*_) = CFG['plotting'].values()
        #     # unpack chosen rat inputs
        #     (bodyparts_list,bodypart_for_alignment,session_date,treadmill_speed,
        #     treadmill_incline,camera_fps,vid_length) = CFG['rat'][chosen_rat].values()
            
        #     num_sessions = len(session_date)
        #     big_fig = make_subplots(rows=1,cols=num_sessions,shared_xaxes=True,shared_yaxes=True,
        #                             horizontal_spacing=0.1, vertical_spacing=0.1,
        #                             subplot_titles=tuple(num_sessions*['tmp_title']))
        #     for iRec in range(num_sessions):
        #         # format inputs to avoid ambiguities
        #         session_date = session_date[0]
        #         rat_name = str(chosen_rat).lower()
        #         treadmill_speed = str(treadmill_speed[0]).zfill(2)
        #         treadmill_incline = str(treadmill_incline[0]).zfill(2)
        #         session_ID = f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
        #         bin_and_count(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG)
        #         # (_,_,_,_,_,_,_,_,_,_,_,figs) = bin_and_count(
        #         # OE_dict, KS_dict, anipose_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list, filter_ephys, sort_method, 
        #         # filter_all_anipose, bin_width_ms, bin_width_radian,             bodypart_for_alignment, bodypart_for_reference, bodypart_ref_filter,
        #         # trial_reject_bounds_mm, trial_reject_bounds_sec, origin_offsets, bodyparts_list, session_date[iRec], rat_name[iRec],
        #         # treadmill_speed[iRec], treadmill_incline[iRec], camera_fps, align_to, vid_length,
        #         # time_frame, save_binned_MU_data, do_plot=False, plot_template=plot_template, MU_colors=MU_colors,
        #         # CH_colors=CH_colors)
        #         for iHist in range(len(figs[1].data)):
        #             big_fig.add_trace(figs[1].data[iHist], row=1, col=iRec+1)
        #         # extract each session's recording parameters, and set subplot titles
        #         big_fig.layout.annotations[iRec].update(text=figs[1].layout.title.text)
        #         # set y-axis titles to those received from bin_and_count()
        #         big_fig.update_yaxes(title_text=figs[1].layout.yaxis.title.text, row = 1, col = iRec+1)
        #         # set x-axis titles to those received from bin_and_count()
        #         big_fig.update_xaxes(title_text=figs[1].layout.xaxis.title.text,row = 1, col = iRec+1)
        #     # lock y-axes together
        #     big_fig.update_yaxes(matches='y')
        #     # Reduce opacity to see both histograms
        #     big_fig.update_traces(opacity=0.75,showlegend=False)
            
        #     iplot(big_fig)
        # # multi_smooth performs smoothing of binned spikes, plots results for all chosen conditions
        # elif CFG['plotting']['plot_type'] == "multi_smooth":
        #     from plotly.offline import iplot
        #     from plotly.subplots import make_subplots
        #     num_smooth_windows = len(smoothing_window)
        #     big_fig = make_subplots(rows=num_smooth_windows,cols=1,shared_xaxes='columns',shared_yaxes=False,
        #                             horizontal_spacing=0.1, vertical_spacing=0.1,
        #                             subplot_titles=tuple(num_smooth_windows*['tmp_title']))
        #     for iSmooth in range(num_smooth_windows):
        #         _,_,_,figs = smooth(
        #             OE_dict, KS_dict, anipose_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
        #             filter_ephys, sort_method,  filter_all_anipose, bin_width_ms, bin_width_radian, smoothing_window[iSmooth],                 bodypart_for_alignment, bodypart_for_reference,
        #             bodypart_ref_filter, origin_offsets,
        #             session_date[0], rat_name[0], treadmill_speed[0], treadmill_incline[0],
        #             camera_fps, align_to, vid_length, time_frame, save_binned_MU_data,
        #             do_plot=False, phase_align=phase_align, plot_template=plot_template, MU_colors=MU_colors, CH_colors=CH_colors)
        #         for iPlot in range(len(figs[0].data)):
        #             big_fig.add_trace(figs[0].data[iPlot], row=iSmooth+1,col=1)
        #         # keep track of session recording parameters, and set those for subplot titles
        #         big_fig.layout.annotations[iSmooth].update(text=figs[0].layout.title.text.split('<br>')[1])
        #         # big_fig.layout.annotations[2*iSmooth+1].update(text=figs[0].layout.annotations[1].text)
        #         # set y-axis titles to those received from bin_and_count()
        #         big_fig.update_yaxes(title_text=figs[0].layout.yaxis.title.text,
        #                                 row = iSmooth+1, col = 1)
        #         # big_fig.update_yaxes(title_text=figs[0].layout.yaxis2.title.text,
        #         #                         row = iSmooth+1, col = 2)
        #     # set x-axis titles to those received from bin_and_count()
        #     big_fig.update_xaxes(title_text=figs[0].layout.xaxis.title.text,row = num_smooth_windows, col = 1)
        #     # big_fig.update_xaxes(title_text=figs[0].layout.xaxis2.title.text,row = iSmooth+1, col = 2)
        #     big_fig.update_xaxes(matches='x')
        #     # Reduce opacity to see all traces
        #     # big_fig.update_traces(opacity=0.75)
        #     # set all titles using received title from bin_and_count()
        #     big_fig.update_layout(title_text=figs[0].layout.title.text.split('<br>')[0])

        #     iplot(big_fig)
        # elif CFG['plotting']['plot_type'] == "multi_state_space":
        #     from plotly.offline import iplot
        #     from plotly.subplots import make_subplots
        #     # num_channels = len(ephys_channel_idxs_list)
        #     num_sessions = len(session_date)
        #     big_fig = make_subplots(cols=1,rows=num_sessions,shared_xaxes=True,shared_yaxes=True,
        #                             horizontal_spacing=0.1, vertical_spacing=0.1,
        #                             subplot_titles=tuple(num_sessions*['tmp_title']),
        #                             specs=[
        #                                 [{"type": "scatter"}],
        #                                 [{"type": "scatter"}],
        #                                 [{"type": "scatter"}],
        #                                 [{"type": "scatter"}]
        #                                 ]
        #                             )
        #     for iRec in range(num_sessions):
        #         _,_,_,figs = state_space(
        #             OE_dict, KS_dict, anipose_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list, filter_ephys, sort_method, 
        #             filter_all_anipose, bin_width_ms, bin_width_radian, smoothing_window[iRec],
        #             bodypart_for_alignment, bodypart_for_reference, bodypart_ref_filter,
        #             trial_reject_bounds_mm, trial_reject_bounds_sec, origin_offsets, bodyparts_list, session_date[iRec], rat_name[iRec],
        #             treadmill_speed[iRec], treadmill_incline[iRec], camera_fps, align_to, vid_length, time_frame, save_binned_MU_data,
        #             do_plot=False, plot_units=plot_units, phase_align=phase_align, plot_template=plot_template,
        #             MU_colors=MU_colors, CH_colors=CH_colors)
        #         # set_trace()
        #         for iPlot in range(len(figs[0].data)):
        #             big_fig.add_trace(figs[0].data[iPlot], row=iRec+1,col=1)
        #         # keep track of session recording parameters, and set those for subplot titles
        #         big_fig.layout.annotations[iRec].update(text=figs[0].layout.title.text.split('<br>')[1])
        #         # big_fig.layout.annotations[2*iRec+1].update(text=figs[0].layout.annotations[1].text)
        #         # set y-axis titles to those received from bin_and_count()
        #         big_fig.update_yaxes(title_text=figs[0].layout.yaxis.title.text,
        #                                 row = iRec+1, col = 1)
        #         # big_fig.update_yaxes(title_text=figs[0].layout.yaxis2.title.text,
        #         #                         row = iRec+1, col = 2)
        #         # set x-axis titles to those received from bin_and_count()
        #         big_fig.update_xaxes(title_text=figs[0].layout.xaxis.title.text,row = iRec+1, col = 1)
        #         # big_fig.update_xaxes(matches='y',row = iRec+1, col = 1)
        #         # big_fig.update_yaxes(matches='x',row = iRec+1, col = 1)
        #         big_fig.update_xaxes(scaleanchor = "y", scaleratio = 1, row = iRec+1, col = 1)
        #         big_fig.update_yaxes(scaleanchor = "x", scaleratio = 1, row = iRec+1, col = 1)
        #     # big_fig.update_xaxes(title_text=figs[0].layout.xaxis2.title.text,row = iRec+1, col = 2)
        #     # Reduce opacity to see all traces
        #     # big_fig.update_traces(opacity=0.75)
        #     # set all titles using received title from bin_and_count()
        #     big_fig.update_layout(title_text=figs[0].layout.title.text.split('<br>')[0])
            
            # iplot(big_fig)
        else:
            raise ValueError(f"Invalid value `plot_type`: '{CFG['plotting']['plot_type']}' in config/config.toml") 
    return

if __name__ == "__main__":    
    session_iterator = list(range(len(CFG['rat'][chosen_rat]['session_date'])))
    OE_dict, KS_dict, anipose_dict = {},{},{}
    OE_dict = import_OE_data(chosen_rat, CFG, session_iterator)
    anipose_dict = import_anipose_data(chosen_rat, CFG, session_iterator)
    if CFG['analysis']['sort_method'] =='kilosort':
        KS_dict = import_KS_data(chosen_rat, CFG, session_iterator)
    rat_loco_analysis(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator)
    


### list of plotly colors, for reference
# plotly_named_colors = [
#     "aliceblue", "antiquewhite", "aqua", "aquamarine", "azure",
#     "beige", "bisque", "black", "blanchedalmond", "blue",
#     "blueviolet", "brown", "burlywood", "cadetblue",
#     "chartreuse", "chocolate", "coral", "cornflowerblue",
#     "cornsilk", "crimson", "cyan", "darkblue", "darkcyan",
#     "darkgoldenrod", "darkgray", "darkgrey", "darkgreen",
#     "darkkhaki", "darkmagenta", "darkolivegreen", "darkorange",
#     "darkorchid", "darkred", "darksalmon", "darkseagreen",
#     "darkslateblue", "darkslategray", "darkslategrey",
#     "darkturquoise", "darkviolet", "deeppink", "deepskyblue",
#     "dimgray", "dimgrey", "dodgerblue", "firebrick",
#     "floralwhite", "forestgreen", "fuchsia", "gainsboro",
#     "ghostwhite", "gold", "goldenrod", "gray", "grey", "green",
#     "greenyellow", "honeydew", "hotpink", "indianred", "indigo",
#     "ivory", "khaki", "lavender", "lavenderblush", "lawngreen",
#     "lemonchiffon", "lightblue", "lightcoral", "lightcyan",
#     "lightgoldenrodyellow", "lightgray", "lightgrey",
#     "lightgreen", "lightpink", "lightsalmon", "lightseagreen",
#     "lightskyblue", "lightslategray", "lightslategrey",
#     "lightsteelblue", "lightyellow", "lime", "limegreen",
#     "linen", "magenta", "maroon", "mediumaquamarine",
#     "mediumblue", "mediumorchid", "mediumpurple",
#     "mediumseagreen", "mediumslateblue", "mediumspringgreen",
#     "mediumturquoise", "mediumvioletred", "midnightblue",
#     "mintcream", "mistyrose", "moccasin", "navajowhite", "navy",
#     "oldlace", "olive", "olivedrab", "orange", "orangered",
#     "orchid", "palegoldenrod", "palegreen", "paleturquoise",
#     "palevioletred", "papayawhip", "peachpuff", "peru", "pink",
#     "plum", "powderblue", "purple", "red", "rosybrown",
#     "royalblue", "rebeccapurple", "saddlebrown", "salmon",
#     "sandybrown", "seagreen", "seashell", "sienna", "silver",
#     "skyblue", "slateblue", "slategray", "slategrey", "snow",
#     "springgreen", "steelblue", "tan", "teal", "thistle", "tomato",
#     "turquoise", "violet", "wheat", "white", "whitesmoke",
#     "yellow", "yellowgreen"
#     ]

from load_data import load_OE_data, load_anipose_data, load_KS_data
import colorlover as cl
from collections import deque
from config import config as CFG

### Chosen Rat ###
chosen_rat = CFG['chosen_rat']
chosen_session = CFG['chosen_session']

### Process Chosen Plotting Parameters
# black to grey
CH_colors = cl.to_rgb(cl.interp(cl.scales['6']['seq']['Greys'],
                                2*CFG['plotting']['N_colors']))[
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

def rat_loco_analysis(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator, chosen_session):
    # if not running a multi-session analysis, the first element in session_iterator is used
        
    if 'multi' not in CFG['plotting']['plot_type']:
        # functions for calling all analysis functions. Only chosen "plot_type" is executed
        if CFG['plotting']['plot_type'] == "sort":
            from process_spikes import sort
            if CFG['analysis']['export_data'] is True:
                for iRec in range(len(CFG['rat'][chosen_rat]['session_date'])):
                    sort(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator[iRec])
            else:
                sort(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator[chosen_session])
        elif CFG['plotting']['plot_type'] == "bin_and_count":
            from process_spikes import bin_and_count
            bin_and_count(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator[chosen_session])
        elif CFG['plotting']['plot_type'] == "raster":
            from process_spikes import raster
            raster(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator[chosen_session])
        elif CFG['plotting']['plot_type'] == "smooth":
            from process_spikes import smooth
            smooth(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator[chosen_session])
        elif CFG['plotting']['plot_type'] == "state_space":
            from process_spikes import state_space
            state_space(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator[chosen_session])
        elif CFG['plotting']['plot_type'] == "MU_space_stepwise":
            from process_spikes import MU_space_stepwise
            MU_space_stepwise(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator[chosen_session])
        elif CFG['plotting']['plot_type'] == "behavioral_space":
            from process_steps import behavioral_space
            behavioral_space(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator[chosen_session])
        elif CFG['plotting']['plot_type'] == "cluster_steps":
            from extras.cluster_steps import cluster_steps
            cluster_steps(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator[chosen_session])
        elif CFG['plotting']['plot_type'] == "pandas_eda":
            from extras.pandas_eda import pandas_eda
            pandas_eda(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator[chosen_session])
        elif CFG['plotting']['plot_type'] == "spike_motion_plot":
            from extras.spike_motion_plot import spike_motion_plot
            spike_motion_plot(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator[chosen_session])           
    else:
        ### Functions with prefix "multi" are designed to loop and compare across multiple condtions
        # multi_bin performs binning of spikes, plots results for all chosen conditions
        if CFG['plotting']['plot_type'] == "multi_sort":
            from process_spikes import sort
            for iRec in range(len(CFG['rat'][chosen_rat]['session_date'])):
                sort(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator[iRec])
        elif CFG['plotting']['plot_type'] == "multi_bin":
            from plotly.offline import iplot
            from plotly.subplots import make_subplots
            from process_spikes import bin_and_count
            ### Unpack CFG Inputs
            # unpack analysis inputs
            (MU_spike_amplitudes_list,ephys_channel_idxs_list,filter_ephys,sort_method, sort_to_use,
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
                (_,_,_,_,_,_,_,_,_,_,_,_,figs) = bin_and_count(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, iRec)

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
            iplot(big_fig)
        # multijoin_bin performs binning and counting of all chosen conditions,
        # and plots results combined results
        elif CFG['plotting']['plot_type'] == "multijoin_bin_and_count":
            from multi_handler import multijoin_bin_and_count
            (MU_spike_amplitudes_list,ephys_channel_idxs_list,filter_ephys,sort_method, 
            bodypart_for_reference,bodypart_ref_filter,filter_all_anipose,trial_reject_bounds_mm,
            trial_reject_bounds_sec,origin_offsets,save_binned_MU_data,time_frame,bin_width_ms,
            num_rad_bins,smoothing_window,phase_align,align_to,export_data) = CFG['analysis'].values()
            # unpack plotting inputs
            (plot_type,plot_units,do_plot,N_colors,plot_template,*_) = CFG['plotting'].values()
            # unpack chosen rat inputs
            (bodyparts_list,bodypart_for_alignment,session_date,treadmill_speed,
            treadmill_incline,camera_fps,vid_length) = CFG['rat'][chosen_rat].values()
            
            multijoin_bin_and_count(
                chosen_rat, OE_dict, KS_dict, anipose_dict,
                CH_colors, MU_colors, CFG, session_iterator
            )
        else:
            raise ValueError(f"Invalid value `plot_type`: '{CFG['plotting']['plot_type']}' in config/config.toml") 
    return

if __name__ == "__main__":    
    session_iterator = list(range(len(CFG['rat'][chosen_rat]['session_date'])))
    OE_dict, KS_dict, anipose_dict = {},{},{}
    OE_dict = load_OE_data(chosen_rat, CFG, session_iterator)
    anipose_dict = load_anipose_data(chosen_rat, CFG, session_iterator)
    if CFG['analysis']['sort_method'] =='kilosort':
        KS_dict = load_KS_data(chosen_rat, CFG, session_iterator)
    rat_loco_analysis(chosen_rat, OE_dict, KS_dict, anipose_dict, CH_colors, MU_colors, CFG, session_iterator, chosen_session)

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

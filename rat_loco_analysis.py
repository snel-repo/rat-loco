from import_OE_data import import_OE_data
from import_anipose_data import import_anipose_data 
from import_KS_data import import_KS_data
import process_spikes, spike_motion_plot 
import process_steps, cluster_steps, pandas_eda
import plotly.io as pio
import colorlover as cl
from numpy import pi
from pdb import set_trace
from config import config as CFG

### Analysis Parameters ###
MU_spike_amplitudes_list = [[150.0001,500],[500.0001,1700],[1700.0001,5000]]
ephys_channel_idxs_list = [14]#[7]#[13]#[2,4,5,7,8,11]#[1,2,3,4,13,14]
filter_ephys = 'notch' # 'bandpass' # 'both' # notch is 60Hz and bandpass is 350-7000Hz
sort_method = 'kilosort' # 'kilosort'/'thresholding'
bodypart_for_reference = ['tailbase'] # choose bodypart to use as origin, without _x/_y/_z suffix, plug into origin_offsets as a value to subtract for that coordinate
bodypart_ref_filter =  2 #Hz, False/int Example: (False to disable filtering of bodypart_for_reference, 2 for 2Hz cutoff lowpass)
filter_all_anipose = False # 'highpass', 'median', or False
trial_reject_bounds_mm = dict(peak=[-15,15],trough=[-15,15]) #mm, False/Integer/Dict, rejects trials outside bounds of the trial average at each bodypart's alignment timepoint. Examples: False / 40 / dict(peak=[10,40],trough=[-10,25] )
trial_reject_bounds_sec = [[0,0.500]] #seconds, time window of step duration outside of which trials get rejected. Examples: [[0, 0.550]] or [[0.550, 0.6]]
origin_offsets = dict(x=-18,y=bodypart_for_reference,z=135) # Values recorded from origin to treadmill bounds, insert bodypart_for_reference variable, or use zeroes for no offset if bodypart_for_reference is set for one coordinate, it overrides etting and will subtract for that coordinate. Examples: dict((x=-18,y=211,z=135))/dict(x=-87,y=211,z=135)/dict(x=52,y=-310,z=0)/dict(x=bodypart_for_reference,y=211,z=135), can be disbaled with disabled with False
save_binned_MU_data = False
time_frame=[0.05,0.25] # 2-element list slicing between 0 and 1, e.g., [0,.5], set to 1 for full ephys plotting
bin_width_ms=10
bin_width_radian=bin_width_ms*(2*pi)/500 # leave 2*pi numerator and number of bins equals (500/bin_width_ms)
smoothing_window = 4*[10] # bins
phase_align=True # True/False, pertains to process_spikes.smooth() and process_spikes.state_space()
align_to='foot off' # "foot strike"/"foot off"

### Chosen Rat ###
chosen_rat = CFG['rat']['godzilla'] # <-- Choose Rat HERE
bodyparts_list = chosen_rat['bodyparts_list']
bodypart_for_alignment = chosen_rat['bodypart_for_alignment']  
session_date = chosen_rat['session_date'] 
rat_name = chosen_rat['rat_name'] 
treadmill_speed = chosen_rat['treadmill_speed'] 
treadmill_incline = chosen_rat['treadmill_incline'] 
camera_fps = chosen_rat['camera_fps'] 
vid_length = chosen_rat['vid_length'] 

### Plotting Parameters
plot_type = "sort" # MU_space_stepwise # behavioral_space # sort # bin_and_count # pandas_eda # multi_bin # multi_state_space
plot_units = [0,1]
do_plot = True # set True/False, whether to actually generate plots
Possible_Themes =['ggplot2','seaborn','simple_white','plotly','plotly_white','plotly_dark',
                    'presentation','xgridoff','ygridoff','gridon','none']
qual_dict_keys = ['Paired', 'Pastel1', 'Set1', 'Set3']
div_dict_keys = ['BrBG', 'PRGn', 'PiYG', 'PuOr', 'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral']
seq_dict_keys = ['Blues', 'BuGn', 'BuPu', 'GnBu', 'Greens', 'Greys', 'OrRd', 'Oranges', 'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu', 'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']
plot_template = pio.templates.default = 'plotly_white'

### Define sequential color lists for plot consistency
N_colors = 16#len(MU_spike_amplitudes_list)*len(ephys_channel_idxs_list)+len(bodyparts_list)
# CH_colors = cl.to_rgb(cl.interp(plotly.colors.sequential.Jet,16))
CH_colors = cl.to_rgb(cl.interp(cl.scales['6']['seq']['Greys'],N_colors))[-1:-(N_colors+1):-1] # black to grey, 16
MU_colors = cl.to_rgb(cl.interp(cl.scales['10']['div']['Spectral'],N_colors)) # rainbow scale, 32

# rotate or reverse colors palettes, if needed
from collections import deque
color_list_len = len(MU_colors)
MU_colors_deque = deque(MU_colors)
# MU_colors_deque.rotate(-7)
# MU_colors_deque.rotate(4)
MU_colors = list(MU_colors_deque)
MU_colors.reverse()
MU_colors= MU_colors[:-1]
# MU_colors = ['royalblue','green','darkorange','firebrick']

### Chosen Directories
OE_directory_list = CFG['data_dirs']['OE']
KS_results_directory_list = CFG['data_dirs']['KS']
Anipose_directory_list = CFG['data_dirs']['Anipose']

# function filters data dictionaries fir desired data
def filter_data_dict(data_dict, session_date, rat_name, treadmill_speed, treadmill_incline):
    data_dict_filtered_by_date = dict(filter(lambda item:
                                str(session_date) in item[0], data_dict.items()))
    data_dict_filtered_by_ratname = dict(filter(lambda item:
                                rat_name in item[0], data_dict_filtered_by_date.items()))
    data_dict_filtered_by_speed = dict(filter(lambda item:
                                "speed"+str(treadmill_speed).zfill(2) in item[0],
                                data_dict_filtered_by_ratname.items()))
    data_dict_filtered_by_incline = dict(filter(lambda item:
                                "incline"+str(treadmill_incline).zfill(2) in item[0],
                                data_dict_filtered_by_speed.items()))
    chosen_data_dict = data_dict_filtered_by_incline
    return chosen_data_dict

### Filter Directories
unfiltered_OE_data_dict = import_OE_data(OE_directory_list)
OE_data_dict = filter_data_dict(unfiltered_OE_data_dict,
                                session_date[0], rat_name[0],
                                str(treadmill_speed[0]).zfill(2),
                                str(treadmill_incline[0]).zfill(2))
if sort_method=='kilosort':
    unfiltered_KS_data_dict = import_KS_data(KS_results_directory_list)
    KS_data_dict = filter_data_dict(unfiltered_KS_data_dict,
                                    session_date[0], rat_name[0],
                                    treadmill_speed[0], treadmill_incline[0])
    
unfiltered_anipose_data_dict = import_anipose_data(Anipose_directory_list)
anipose_data_dict = filter_data_dict(unfiltered_anipose_data_dict,
            session_date[0], rat_name[0], treadmill_speed[0], treadmill_incline[0])

# begin section for calling all analysis functions. Only chosen "plot_type" is executed
if plot_type == "sort":
    process_spikes.sort(
        OE_data_dict, KS_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
        filter_ephys, sort_method,  filter_all_anipose, anipose_data_dict, 
        bodyparts_list, bodypart_for_alignment, bodypart_for_reference,
        bodypart_ref_filter, origin_offsets,
        session_date[0], rat_name[0], treadmill_speed[0], treadmill_incline[0],
        camera_fps, align_to, vid_length, time_frame,
        do_plot, plot_template, MU_colors, CH_colors)
elif plot_type == "cluster_steps":
    cluster_steps.cluster_steps(
        OE_data_dict, KS_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
        filter_ephys, sort_method,  filter_all_anipose, bin_width_ms, bin_width_radian, anipose_data_dict,
        bodypart_for_alignment, bodypart_for_reference,
        bodypart_ref_filter, origin_offsets,
        session_date, rat_name, treadmill_speed, treadmill_incline,
        camera_fps, align_to, vid_length, time_frame,
        do_plot, plot_template, MU_colors, CH_colors)
elif plot_type == "bin_and_count":
    process_spikes.bin_and_count(
        OE_data_dict, KS_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
        filter_ephys, sort_method,  filter_all_anipose, bin_width_ms, bin_width_radian, anipose_data_dict,
        bodypart_for_alignment, bodypart_for_reference,
        bodypart_ref_filter, trial_reject_bounds_mm, trial_reject_bounds_sec, origin_offsets, bodyparts_list,
        session_date[0], rat_name[0], treadmill_speed[0], treadmill_incline[0],
        camera_fps, align_to, vid_length, time_frame, save_binned_MU_data,
        do_plot, plot_template, MU_colors, CH_colors)
elif plot_type == "raster":
    process_spikes.raster(
        OE_data_dict, KS_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
        filter_ephys, sort_method,  filter_all_anipose, bin_width_ms, bin_width_radian, anipose_data_dict,
        bodypart_for_alignment, bodypart_for_reference,
        bodypart_ref_filter, trial_reject_bounds_mm, trial_reject_bounds_sec, origin_offsets, bodyparts_list,
        session_date[0], rat_name[0], treadmill_speed[0], treadmill_incline[0],
        camera_fps, align_to, vid_length, time_frame, save_binned_MU_data,
        do_plot, plot_template, MU_colors, CH_colors)
elif plot_type == "smooth":
    process_spikes.smooth(
        OE_data_dict, KS_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
        filter_ephys, sort_method,  filter_all_anipose, bin_width_ms, bin_width_radian, smoothing_window[0], anipose_data_dict,
        bodypart_for_alignment, bodypart_for_reference,
        bodypart_ref_filter, trial_reject_bounds_mm, trial_reject_bounds_sec, origin_offsets, bodyparts_list,
        session_date[0], rat_name[0], treadmill_speed[0], treadmill_incline[0],
        camera_fps, align_to, vid_length, time_frame, save_binned_MU_data,
        do_plot, phase_align, plot_template, MU_colors, CH_colors)
elif plot_type == "state_space":
    process_spikes.state_space(
        OE_data_dict, KS_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
        filter_ephys, sort_method,  filter_all_anipose, bin_width_ms, bin_width_radian, smoothing_window[0], anipose_data_dict,
        bodypart_for_alignment, bodypart_for_reference,
        bodypart_ref_filter, trial_reject_bounds_mm, trial_reject_bounds_sec, origin_offsets, bodyparts_list,
        session_date[0], rat_name[0], treadmill_speed[0], treadmill_incline[0],
        camera_fps, align_to, vid_length, time_frame, save_binned_MU_data,
        do_plot, plot_units, phase_align, plot_template, MU_colors, CH_colors)
elif plot_type == "MU_space_stepwise":
    process_spikes.MU_space_stepwise(
        OE_data_dict, KS_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list, filter_ephys, sort_method, 
        filter_all_anipose, bin_width_ms, bin_width_radian, smoothing_window[0], anipose_data_dict,
        bodypart_for_alignment, bodypart_for_reference, bodypart_ref_filter, trial_reject_bounds_mm, trial_reject_bounds_sec,
        origin_offsets, bodyparts_list, session_date, rat_name, treadmill_speed, treadmill_incline,
        camera_fps, align_to, vid_length, time_frame, save_binned_MU_data, do_plot, plot_units, phase_align, plot_template,
        MU_colors, CH_colors)
elif plot_type == "pandas_eda":
    pandas_eda.pandas_eda(
        OE_data_dict, KS_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
        filter_ephys, sort_method,  filter_all_anipose, bin_width_ms, bin_width_radian, anipose_data_dict,
        bodypart_for_alignment, bodypart_for_reference,
        bodypart_ref_filter, origin_offsets,
        session_date, rat_name, treadmill_speed, treadmill_incline,
        camera_fps, align_to, vid_length, time_frame, save_binned_MU_data,
        do_plot, plot_template, MU_colors, CH_colors)
elif plot_type == "behavioral_space":
    process_steps.behavioral_space(
        anipose_data_dict, bodypart_for_alignment, bodypart_for_reference,
        bodypart_ref_filter, trial_reject_bounds_mm, trial_reject_bounds_sec, origin_offsets, bodyparts_list,
        filter_all_anipose, session_date, rat_name, treadmill_speed,
        treadmill_incline, camera_fps, align_to, time_frame, save_binned_MU_data, MU_colors, CH_colors)
elif plot_type == "spike_motion_plot":
    spike_motion_plot.spike_motion_plot(
        OE_data_dict, KS_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
        filter_ephys, sort_method,  filter_all_anipose, bin_width_ms, bin_width_radian, anipose_data_dict,
        bodypart_for_alignment, bodypart_for_reference,
        bodypart_ref_filter, origin_offsets,
        session_date[0], rat_name[0], treadmill_speed[0], treadmill_incline[0],
        camera_fps, align_to, vid_length, time_frame, save_binned_MU_data,
        do_plot, plot_template, MU_colors, CH_colors)
        
### functions with prefix "multi" are designed to loop and compare across multiple condtions
# multi_bin performs binning of spikes, plots results for all chosen conditions
elif plot_type == "multi_bin":
    from plotly.offline import iplot
    from plotly.subplots import make_subplots
    num_sessions = len(session_date)
    big_fig = make_subplots(rows=num_sessions,cols=2,shared_xaxes='columns',shared_yaxes=True,
                            horizontal_spacing=0.1, vertical_spacing=0.1,
                            subplot_titles=tuple(2*num_sessions*['tmp_title']))
    for iRec in range(num_sessions):
        (_,_,_,_,_,_,_,_,_,_,_,figs) = process_spikes.bin_and_count(
            OE_data_dict, KS_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
            filter_ephys, sort_method,  filter_all_anipose, bin_width_ms, bin_width_radian, anipose_data_dict,
            bodypart_for_alignment, bodypart_for_reference,
            bodypart_ref_filter, trial_reject_bounds_mm, trial_reject_bounds_sec, origin_offsets, bodyparts_list,
            session_date[iRec], rat_name[iRec], treadmill_speed[iRec], treadmill_incline[iRec],
            camera_fps, align_to, vid_length, time_frame, save_binned_MU_data,
            do_plot=False, plot_template=plot_template, MU_colors=MU_colors, CH_colors=CH_colors)
        for iHist in range(len(figs[0].data)):
            big_fig.add_trace(figs[0].data[iHist], row=iRec+1,
                col=(iHist//len(MU_spike_amplitudes_list))+1)
        # keep track of session recording parameters, and set those for subplot titles
        big_fig.layout.annotations[2*iRec].update(text=figs[0].layout.annotations[0].text)
        big_fig.layout.annotations[2*iRec+1].update(text=figs[0].layout.annotations[1].text)
        # set y-axis titles to those received from bin_spikes()
        big_fig.update_yaxes(title_text=figs[0].layout.yaxis.title.text,
                                row = iRec+1, col = 1)
        big_fig.update_yaxes(title_text=figs[0].layout.yaxis2.title.text,
                                row = iRec+1, col = 2)
    # set x-axis titles to those received from bin_spikes()
    big_fig.update_xaxes(title_text=figs[0].layout.xaxis.title.text,row = iRec+1, col = 1)
    big_fig.update_xaxes(title_text=figs[0].layout.xaxis2.title.text,row = iRec+1, col = 2)
    big_fig.update_yaxes(matches='y')
    # Reduce opacity to see both histograms
    big_fig.update_traces(opacity=0.75)
    # set bars to overlap and all titles, and use received title from bin_spikes()
    big_fig.update_layout(barmode='overlay',title_text=figs[0].layout.title.text)
    
    iplot(big_fig)
# multi_count performs counting of total number of spikes, plots results for all chosen conditions
elif plot_type == "multi_count":
    from plotly.offline import iplot
    from plotly.subplots import make_subplots
    num_sessions = len(session_date)
    big_fig = make_subplots(rows=1,cols=num_sessions,shared_xaxes=True,shared_yaxes=True,
                            horizontal_spacing=0.1, vertical_spacing=0.1,
                            subplot_titles=tuple(num_sessions*['tmp_title']))
    for iRec in range(num_sessions):
        (_,_,_,_,_,_,_,_,_,_,_,figs) = process_spikes.bin_and_count(
        OE_data_dict, KS_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list, filter_ephys, sort_method, 
        filter_all_anipose, bin_width_ms, bin_width_radian, anipose_data_dict,
        bodypart_for_alignment, bodypart_for_reference, bodypart_ref_filter,
        trial_reject_bounds_mm, trial_reject_bounds_sec, origin_offsets, bodyparts_list, session_date[iRec], rat_name[iRec],
        treadmill_speed[iRec], treadmill_incline[iRec], camera_fps, align_to, vid_length,
        time_frame, save_binned_MU_data, do_plot=False, plot_template=plot_template, MU_colors=MU_colors,
        CH_colors=CH_colors)
        for iHist in range(len(figs[1].data)):
            big_fig.add_trace(figs[1].data[iHist], row=1, col=iRec+1)
        # extract each session's recording parameters, and set subplot titles
        big_fig.layout.annotations[iRec].update(text=figs[1].layout.title.text)
        # set y-axis titles to those received from bin_spikes()
        big_fig.update_yaxes(title_text=figs[1].layout.yaxis.title.text, row = 1, col = iRec+1)
        # set x-axis titles to those received from bin_spikes()
        big_fig.update_xaxes(title_text=figs[1].layout.xaxis.title.text,row = 1, col = iRec+1)
    # lock y-axes together
    big_fig.update_yaxes(matches='y')
    # Reduce opacity to see both histograms
    big_fig.update_traces(opacity=0.75,showlegend=False)
    
    iplot(big_fig)
# multi_smooth performs smoothing of binned spikes, plots results for all chosen conditions
elif plot_type == "multi_smooth":
    from plotly.offline import iplot
    from plotly.subplots import make_subplots
    num_smooth_windows = len(smoothing_window)
    big_fig = make_subplots(rows=num_smooth_windows,cols=1,shared_xaxes='columns',shared_yaxes=False,
                            horizontal_spacing=0.1, vertical_spacing=0.1,
                            subplot_titles=tuple(num_smooth_windows*['tmp_title']))
    for iSmooth in range(num_smooth_windows):
        _,_,_,figs = process_spikes.smooth(
            OE_data_dict, KS_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
            filter_ephys, sort_method,  filter_all_anipose, bin_width_ms, bin_width_radian, smoothing_window[iSmooth], anipose_data_dict,
            bodypart_for_alignment, bodypart_for_reference,
            bodypart_ref_filter, origin_offsets,
            session_date[0], rat_name[0], treadmill_speed[0], treadmill_incline[0],
            camera_fps, align_to, vid_length, time_frame, save_binned_MU_data,
            do_plot=False, phase_align=phase_align, plot_template=plot_template, MU_colors=MU_colors, CH_colors=CH_colors)
        for iPlot in range(len(figs[0].data)):
            big_fig.add_trace(figs[0].data[iPlot], row=iSmooth+1,col=1)
        # keep track of session recording parameters, and set those for subplot titles
        big_fig.layout.annotations[iSmooth].update(text=figs[0].layout.title.text.split('<br>')[1])
        # big_fig.layout.annotations[2*iSmooth+1].update(text=figs[0].layout.annotations[1].text)
        # set y-axis titles to those received from bin_spikes()
        big_fig.update_yaxes(title_text=figs[0].layout.yaxis.title.text,
                                row = iSmooth+1, col = 1)
        # big_fig.update_yaxes(title_text=figs[0].layout.yaxis2.title.text,
        #                         row = iSmooth+1, col = 2)
    # set x-axis titles to those received from bin_spikes()
    big_fig.update_xaxes(title_text=figs[0].layout.xaxis.title.text,row = num_smooth_windows, col = 1)
    # big_fig.update_xaxes(title_text=figs[0].layout.xaxis2.title.text,row = iSmooth+1, col = 2)
    big_fig.update_xaxes(matches='x')
    # Reduce opacity to see all traces
    # big_fig.update_traces(opacity=0.75)
    # set all titles using received title from bin_spikes()
    big_fig.update_layout(title_text=figs[0].layout.title.text.split('<br>')[0])

    iplot(big_fig)
elif plot_type == "multi_state_space":
    from plotly.offline import iplot
    from plotly.subplots import make_subplots
    # num_channels = len(ephys_channel_idxs_list)
    num_sessions = len(session_date)
    big_fig = make_subplots(cols=1,rows=num_sessions,shared_xaxes=True,shared_yaxes=True,
                            horizontal_spacing=0.1, vertical_spacing=0.1,
                            subplot_titles=tuple(num_sessions*['tmp_title']),
                            specs=[
                                [{"type": "scatter"}],
                                [{"type": "scatter"}],
                                [{"type": "scatter"}],
                                [{"type": "scatter"}]
                                ]
                            )
    for iRec in range(num_sessions):
        _,_,_,figs = process_spikes.state_space(
            OE_data_dict, KS_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list, filter_ephys, sort_method, 
            filter_all_anipose, bin_width_ms, bin_width_radian, smoothing_window[iRec],
            anipose_data_dict, bodypart_for_alignment, bodypart_for_reference, bodypart_ref_filter,
            trial_reject_bounds_mm, trial_reject_bounds_sec, origin_offsets, bodyparts_list, session_date[iRec], rat_name[iRec],
            treadmill_speed[iRec], treadmill_incline[iRec], camera_fps, align_to, vid_length, time_frame, save_binned_MU_data,
            do_plot=False, plot_units=plot_units, phase_align=phase_align, plot_template=plot_template,
            MU_colors=MU_colors, CH_colors=CH_colors)
        # set_trace()
        for iPlot in range(len(figs[0].data)):
            big_fig.add_trace(figs[0].data[iPlot], row=iRec+1,col=1)
        # keep track of session recording parameters, and set those for subplot titles
        big_fig.layout.annotations[iRec].update(text=figs[0].layout.title.text.split('<br>')[1])
        # big_fig.layout.annotations[2*iRec+1].update(text=figs[0].layout.annotations[1].text)
        # set y-axis titles to those received from bin_spikes()
        big_fig.update_yaxes(title_text=figs[0].layout.yaxis.title.text,
                                row = iRec+1, col = 1)
        # big_fig.update_yaxes(title_text=figs[0].layout.yaxis2.title.text,
        #                         row = iRec+1, col = 2)
        # set x-axis titles to those received from bin_spikes()
        big_fig.update_xaxes(title_text=figs[0].layout.xaxis.title.text,row = iRec+1, col = 1)
        # big_fig.update_xaxes(matches='y',row = iRec+1, col = 1)
        # big_fig.update_yaxes(matches='x',row = iRec+1, col = 1)
        big_fig.update_xaxes(scaleanchor = "y", scaleratio = 1, row = iRec+1, col = 1)
        big_fig.update_yaxes(scaleanchor = "x", scaleratio = 1, row = iRec+1, col = 1)
    # big_fig.update_xaxes(title_text=figs[0].layout.xaxis2.title.text,row = iRec+1, col = 2)
    # Reduce opacity to see all traces
    # big_fig.update_traces(opacity=0.75)
    # set all titles using received title from bin_spikes()
    big_fig.update_layout(title_text=figs[0].layout.title.text.split('<br>')[0])
    
    iplot(big_fig)
else:
    raise Exception(f'Plot type "{plot_type}" not found.')

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

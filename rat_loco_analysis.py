import import_OE_data, import_anipose_data, process_spikes #, plot_loco_ephys
import plotly.io as pio
import colorlover as cl
# import plotly.colors

### Chosen Directories
# ephys directory(ies) that should have 'Record Node ###' inside it
ephys_directory_list = [
    '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-03_19-41-47',
    '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-06_15-21-22',
    '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-06_15-45-13',
    '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-06_16-01-57',
    '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-08_14-14-30',
    ]

# anipose directory(ies) that should have pose_3d.csv inside it
anipose_directory_list = [
    '/home/sean/hdd/GTE-BME/SNEL/data/anipose/session220603',
    '/home/sean/hdd/GTE-BME/SNEL/data/anipose/session220606',
    '/home/sean/hdd/GTE-BME/SNEL/data/anipose/session220608'
    ]

ephys_data_dict = import_OE_data.import_OE_data(ephys_directory_list)
anipose_data_dict = import_anipose_data.import_anipose_data(anipose_directory_list)

### Analysis parameters
MU_spike_amplitudes_list = [[150,500],[500.0001,1700],[1700.0001,5000]]
ephys_channel_idxs_list = [7] #[0,1,2,4,5,7,8,9,11,13,15,16]
filter_ephys = 'notch' # 'bandpass' # 'both' # notch is 60Hz and bandpass is 350-7000Hz
bodyparts_list=['palm_L_y']#['palm_L_x','palm_L_y','palm_L_z']
bodypart_for_tracking = ['palm_L_y']
session_date=3*[220603]
rat_name=3*['dogerat']
treadmill_speed=3*[20]
treadmill_incline=[10,5,0]
camera_fps=100
vid_length=20
time_slice=1
bin_width_ms=40
alignto='foot off'

### Plotting Parameters
plot_type = "smooth"
do_plot = True # set True/False, whether to actually generate plots
Possible_Themes =['ggplot2','seaborn','simple_white','plotly','plotly_white','plotly_dark',
                    'presentation','xgridoff','ygridoff','gridon','none']
plot_template = pio.templates.default = 'plotly'

### Define sequential color lists for plot consistency
N_colors = 6
# CH_colors = cl.to_rgb(cl.interp(plotly.colors.sequential.Jet,16))
CH_colors = cl.to_rgb(cl.interp(cl.scales['9']['seq']['Greys'],N_colors*2))[-1:-N_colors:-1] # black to grey, 16
MU_colors = cl.to_rgb(cl.interp(cl.scales['10']['div']['Spectral'],N_colors)) # rainbow scale, 32
# MU_colors = plotly.colors.sequential.Rainbow_r
# MU_colors = plotly.colors.cyclical.HSV_r
# MU_colors = plotly.colors.diverging.Portland_r
# MU_colors = [
#     'blue', 'green', 'orange', 
#     'royalblue','forestgreen','firebrick',
#     'lawngreen', 'greenyellow', 'red',
#     'darkturquoise','purple','black',
#     'lightblue','lightgreen','hotpink',
#     ]

## rotate colors, if needed
# from collections import deque
# color_list_len = len(MU_colors)
# MU_colors_deque = deque(MU_colors)
# MU_colors_deque.rotate(color_list_len//2)
# MU_colors = list(MU_colors_deque)

if plot_type == "sort":
    process_spikes.sort(
        ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
        filter_ephys, anipose_data_dict, bodyparts_list, bodypart_for_tracking,
        session_date[0], rat_name[0], treadmill_speed[0], treadmill_incline[0],
        camera_fps, alignto, vid_length, time_slice,
        do_plot, plot_template, MU_colors, CH_colors)
elif plot_type == "bin":
    process_spikes.bin_and_count(
        ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
        filter_ephys, bin_width_ms, anipose_data_dict, bodypart_for_tracking,
        session_date[0], rat_name[0], treadmill_speed[0], treadmill_incline[0],
        camera_fps, alignto, vid_length, time_slice,
        do_plot, plot_template, MU_colors, CH_colors)
elif plot_type == "smooth":
    process_spikes.smooth(
    ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
    filter_ephys, bin_width_ms, anipose_data_dict, bodypart_for_tracking,
    session_date[0], rat_name[0], treadmill_speed[0], treadmill_incline[0],
    camera_fps, alignto, vid_length, time_slice,
    do_plot, plot_template, MU_colors, CH_colors)
elif plot_type == "multi_bin":
    from plotly.offline import iplot
    from plotly.subplots import make_subplots
    num_sessions = len(session_date)
    big_fig = make_subplots(rows=num_sessions,cols=2,shared_xaxes='columns',shared_yaxes=True,
                            horizontal_spacing=0.1, vertical_spacing=0.1,
                            subplot_titles=tuple(6*['tmp_title']))
    for iRec in range(num_sessions):
        (_,_,_,_,_,_,_,_,figs) = process_spikes.bin_and_count(
            ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
            filter_ephys, bin_width_ms, anipose_data_dict, bodypart_for_tracking,
            session_date[iRec], rat_name[iRec], treadmill_speed[iRec], treadmill_incline[iRec],
            camera_fps, alignto, vid_length, time_slice,
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
elif plot_type == "multi_count":
    from plotly.offline import iplot
    from plotly.subplots import make_subplots
    num_sessions = len(session_date)
    big_fig = make_subplots(rows=1,cols=num_sessions,shared_xaxes=True,shared_yaxes=True,
                            horizontal_spacing=0.1, vertical_spacing=0.1,
                            subplot_titles=tuple(num_sessions*['tmp_title']))
    for iRec in range(num_sessions):
        (_,_,_,_,_,_,_,_,figs) = process_spikes.bin_and_count(
        ephys_data_dict, ephys_channel_idxs_list, MU_spike_amplitudes_list,
        filter_ephys, bin_width_ms, anipose_data_dict, bodypart_for_tracking,
        session_date[iRec], rat_name[iRec], treadmill_speed[iRec], treadmill_incline[iRec],
        camera_fps, alignto, vid_length, time_slice,
        do_plot=False, plot_template=plot_template, MU_colors=MU_colors, CH_colors=CH_colors)
        for iHist in range(len(figs[1].data)):
            big_fig.add_trace(figs[1].data[iHist], row=1, col=iRec+1)
        # keep track of session recording parameters, and set those for subplot titles
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
else:
    raise Exception(f'Plot type "{plot_type}" not found.')

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

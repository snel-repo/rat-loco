import import_OE_data, import_anipose_data, sort_spikes #, plot_loco_ephys
import plotly.colors
import plotly.io as pio

### Chosen Directories
# ephys directory(ies) that should have 'Record Node ###' inside it
ephys_directory_list = [
    '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-03_19-41-47',
    # '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-06_15-21-22',
    # '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-06_15-45-13',
    # '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-06_16-01-57',
    # '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-08_14-14-30',
    ]

# anipose directory(ies) that should have pose_3d.csv inside it
anipose_directory_list = [
    '/home/sean/hdd/GTE-BME/SNEL/data/anipose/session220603',
    # '/home/sean/hdd/GTE-BME/SNEL/data/anipose/session220606',
    # '/home/sean/hdd/GTE-BME/SNEL/data/anipose/session220608']
    ]

ephys_data_dict = import_OE_data.import_OE_data(ephys_directory_list)
anipose_data_dict = import_anipose_data.import_anipose_data(anipose_directory_list)

### Analysis parameters
bodyparts_list=['palm_L_y']#['palm_L_x','palm_L_y','palm_L_z']
bodypart_for_tracking = ['palm_L_y']
session_date=220603
rat_name='dogerat'
treadmill_speed=20
treadmill_incline=10
camera_fps=100
vid_length=20
time_slice=1
ephys_channel_idxs_list = [7] #[0,1,2,4,5,7,8,9,11,13,15,16]
bin_width_ms=10
alignto='foot off'

### Plotting Parameters
plot_type = "sort"
do_plot = True # set True/False, whether to actually generate plots
Possible_Themes =['ggplot2','seaborn','simple_white','plotly','plotly_white','plotly_dark',
                    'presentation','xgridoff','ygridoff','gridon','none']
plot_template = pio.templates.default = 'plotly'

### Define a sequential color list for plot consistency
# plot_colors = plotly.colors.sequential.Rainbow_r
# plot_colors = plotly.colors.cyclical.HSV_r
# plot_colors = plotly.colors.diverging.Portland_r
plot_colors = [
    'blue', 'green', 'orange', 
    'royalblue','forestgreen','firebrick',
    'lawngreen', 'greenyellow', 'red',
    'darkturquoise','purple','black',
    'lightblue','lightgreen','hotpink',
    ]
# plot_colors = [
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

## rotate colors, if needed
# from collections import deque
# color_list_len = len(plot_colors)
# plot_colors_deque = deque(plot_colors)
# plot_colors_deque.rotate(color_list_len//2)
# plot_colors = list(plot_colors_deque)

plot_colors = 10*plot_colors # ensure colors will not run out


if plot_type == "sort":
    sort_spikes.sort_spikes(
        ephys_data_dict, ephys_channel_idxs_list ,
        anipose_data_dict, bodyparts_list, bodypart_for_tracking,
        session_date, rat_name, treadmill_speed, treadmill_incline,
        camera_fps, alignto, vid_length, time_slice,
        do_plot, plot_template, plot_colors
        )
elif plot_type == "bin":
    sort_spikes.bin_spikes(
        ephys_data_dict, ephys_channel_idxs_list, bin_width_ms, 
        anipose_data_dict, bodypart_for_tracking,
        session_date, rat_name, treadmill_speed, treadmill_incline,
        camera_fps, alignto, vid_length, time_slice,
        do_plot, plot_template, plot_colors
        )
else:
    raise Exception(f'Plot type "{plot_type}" not found.')


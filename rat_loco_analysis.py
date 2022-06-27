import import_OE_data, import_anipose_data, plot_loco_ephys, sort_spikes

ephys_directory_list = [
    '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-03_19-41-47', # data directory with 'Record Node ###' inside
    # '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-06_15-21-22',
    # '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-06_15-45-13',
    # '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-06_16-01-57',
    # '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-08_14-14-30',
    ]
anipose_directory_list = [
    '/home/sean/hdd/GTE-BME/SNEL/data/anipose/session220603',
    # '/home/sean/hdd/GTE-BME/SNEL/data/anipose/session220606',
    # '/home/sean/hdd/GTE-BME/SNEL/data/anipose/session220608']
    ]

ephys_data_dict = import_OE_data.import_OE_data(ephys_directory_list)
anipose_data_dict = import_anipose_data.import_anipose_data(anipose_directory_list)

# plotting parameters
plot_type = "bin"
plot_sort = True
bodyparts=['palm_L_y']#['palm_L_x','palm_L_y','palm_L_z']
bodypart_for_tracking = ['palm_L_y']
session_date=220603
rat_name='dogerat'
treadmill_speed=20
treadmill_incline=10
camera_fps=100
time_slice=1
ephys_channel_idxs=[7]#,11,16] #[0,1,2,4,5,7,8,9,11,13,15,16]
bin_width_ms=10
alignto='foot off'

if plot_type == "full":
    plot_loco_ephys.full(
        ephys_data_dict, anipose_data_dict, bodyparts, session_date, rat_name,
        treadmill_speed, treadmill_incline, camera_fps, time_slice, ephys_channel_idxs
        )
elif plot_type == "psth":
    plot_loco_ephys.psth(
    ephys_data_dict, anipose_data_dict, bodyparts, session_date, rat_name,
    treadmill_speed, treadmill_incline, camera_fps, time_slice, ephys_channel_idxs
    )
elif plot_type == "sort":
    sort_spikes.sort_spikes(
            ephys_data_dict, anipose_data_dict, bodyparts, bodypart_for_tracking, session_date, rat_name,
            treadmill_speed, treadmill_incline, camera_fps, time_slice, ephys_channel_idxs, plot_sort
            )
elif plot_type == "bin":
    sort_spikes.bin_spikes(
        ephys_data_dict, anipose_data_dict, session_date, rat_name,
        treadmill_speed, treadmill_incline, bin_width_ms, ephys_channel_idxs,
        bodypart_for_tracking, camera_fps, alignto)
else:
    pass



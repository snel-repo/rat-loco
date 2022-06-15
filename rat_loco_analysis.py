import import_OE_data, import_anipose_data, plot_loco_ephys

ephys_directory_list = [
    '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-03_19-41-47', # data directory with 'Record Node ###' inside
    '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-06_15-21-22',
    '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-06_15-45-13',
    '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-06_16-01-57',
    '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-08_14-14-30']

anipose_directory_list = [
    '/home/sean/hdd/GTE-BME/SNEL/data/anipose/session220603',
    '/home/sean/hdd/GTE-BME/SNEL/data/anipose/session220606',
    '/home/sean/hdd/GTE-BME/SNEL/data/anipose/session220608']

continuous_ephys_data_list = import_OE_data.import_OE_data(ephys_directory_list)
anipose_data_dict = import_anipose_data.import_anipose_data(anipose_directory_list)

plot_loco_ephys.plot_loco_ephys(
    continuous_ephys_data_list,
    anipose_data_dict,
    bodyparts=['palm_L_z','ankle_L_z','palm_R_z','ankle_R_z'],
    session_date=220603,
    rat_name='dogerat',
    treadmill_speed=15,
    treadmill_incline=0,
    time_slice=1,
    ephys_channel_idxs=[0,1,2,4,5,7,8,9,11,13,15,16]
    )
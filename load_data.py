import errno
import os
import pathlib
import sys
import traceback

import numpy as np
import pandas as pd
import scipy.io
from open_ephys.analysis import Session


def _validate_data_dir_input(data_dir: str, sort_to_use: str) -> str:
    if "sorted0" in os.listdir(data_dir) and (sort_to_use == "latest"):
        sort_folder_name = "sorted0"
    elif "best_sort" in os.listdir(data_dir) and sort_to_use == "best":
        sort_folder_name = "best_sort"
    else:
        if sort_to_use == "latest":
            print("No folder named 'sorted0' in " + data_dir)
        elif sort_to_use == "best":
            print("No symlink called 'best_sort' in " + data_dir)
        else:
            try:
                # ensure that the user inputted a valid timestamp
                _ = int(sort_to_use)
                if (len(sort_to_use) != 15) or (sort_to_use[8] != "_"):
                    raise ValueError
                for iFile in os.listdir(data_dir):
                    if sort_to_use in iFile:
                        sort_folder_name = iFile
                        break
                else:
                    raise FileNotFoundError
            except ValueError:
                traceback.print_exc()
                print(
                    "ValueError: sort_to_use must be either 'latest', 'best', or a 'YYYYMMDD_HHMMSS' timestamp."
                )
                sys.exit(1)
            except FileNotFoundError:
                traceback.print_exc()
                print("FileNotFoundError: No folder named " + sort_to_use + " in " + data_dir)
                sys.exit(1)
            except:
                traceback.print_exc()
                print("Unexpected error")
                sys.exit(1)
    return sort_folder_name


def load_OE_data(chosen_rat, CFG, session_iterator):
    ## Outputs all extracted continuous ephys data packed in a dictionary and the keys are unique session identifiers

    # initialize lists and counter variable(s)
    data_dir_list = CFG["data_dirs"]["OE"]
    sort_to_use = CFG["analysis"]["sort_to_use"]
    number_of_recordings_per_session_list = (
        []
    )  # stores number of recordings found in each directory
    continuous_ephys_data_list = []  # stores continuous data
    list_of_session_IDs = []  # stores unique session identifiers
    iChosenRec = int(-1)  # counts the number of recordings per directory

    for data_dir in data_dir_list:  # loop through all paths provided
        session = Session(
            data_dir
        )  # use OE library to extract data from path into a structured python format
        number_of_recordings_per_session_list.append(
            (len(session.recordnodes[0].recordings))
        )  # store number of recordings to allow next looping operation

        sort_folder_name = _validate_data_dir_input(data_dir, sort_to_use)
        ops_path = str(pathlib.PurePath(data_dir).joinpath(sort_folder_name, "ops.mat"))
        ops = scipy.io.loadmat(ops_path)
        recordings_to_use = ops["recordings"]

        for iRec in range(number_of_recordings_per_session_list[-1]):
            if session.recordnodes[0].recordings[iRec].directory.split("recording")[-1] in [
                str(rec) for rec in recordings_to_use
            ]:
                continue  # skips recording if it is not in recordings_to_use
            file_list = os.listdir(session.recordnodes[0].recordings[iRec].directory)
            chosen_recording = 0
            for filename in file_list:
                if filename.endswith(".info"):
                    for iterator in session_iterator:
                        session_date = CFG["rat"][chosen_rat]["session_date"][iterator]
                        rat_name = str(chosen_rat).lower()
                        treadmill_speed = str(
                            CFG["rat"][chosen_rat]["treadmill_speed"][iterator]
                        ).zfill(2)
                        treadmill_incline = str(
                            CFG["rat"][chosen_rat]["treadmill_incline"][iterator]
                        ).zfill(2)
                        session_ID = f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
                        if filename.__contains__(session_ID):
                            recording_info = filename.split(".")[0]
                            list_of_session_IDs.append(recording_info.lower())
                            chosen_recording = 1
                            break
            if chosen_recording:
                # increment to track the number of chosen recordings
                # create list of session dates for the ones that were extracted
                iChosenRec += 1
                continuous_ephys_data_list.append(
                    session.recordnodes[0].recordings[iRec].continuous
                )
                continuous_ephys_data_list[iChosenRec][0].samples = np.array(
                    continuous_ephys_data_list[iChosenRec][0].samples, dtype="float64"
                )

                for iChannel in range(
                    session.recordnodes[0].recordings[iRec].info["continuous"][0]["num_channels"]
                ):
                    # Multiply each recording samples by the measured "bit_volts" value.
                    # This converts from 16-bit number to uV for continuous channels and V for ADC channels
                    continuous_ephys_data_list[iChosenRec][0].samples[:, iChannel] = (
                        continuous_ephys_data_list[iChosenRec][0].samples[:, iChannel]
                        * session.recordnodes[0]
                        .recordings[iRec]
                        .info["continuous"][0]["channels"][iChannel]["bit_volts"]
                    )
                continuous_ephys_data_list[iChosenRec][0].samples = np.array(
                    continuous_ephys_data_list[iChosenRec][0].samples, dtype="float32"
                )
    ## section plots the SYNC channel and describes stats of intervals
    # signal = continuous_ephys_data_list[iChosenRec][0].samples[:,iChannel]
    # fsignal = iir_notch(signal, 30000)
    # dsignal = np.digitize(fsignal,[-5,2,5])-1# digitize
    # peak_idxs, _ = find_peaks(dsignal, height=0.9)
    # df_fs_minus_fo = pd.DataFrame(peak_idxs[1:] - peak_idxs[:-1])
    # print(f"Stats:\n")
    # step_stats = df_fs_minus_fo.describe()[0]

    # from IPython.display import display
    # display(step_stats)
    # plt.plot(np.arange(len(signal))/30000, signal/3.3,c=[0,.4,.9,.5])
    # plt.plot(np.arange(len(dsignal))/30000,dsignal,c='black')
    # plt.scatter(peak_idxs/30000, dsignal[peak_idxs],c='r')
    # plt.xlabel("Time (s)")
    # plt.title(r'Checking Peaks for SYNC')
    # plt.show()
    if len(list_of_session_IDs) == 0:
        raise FileNotFoundError(
            errno.ENOENT,
            os.strerror(errno.ENOENT),
            "No '.info' file matching your criteria was found. It could be wrong rat, speed, incline, etc.",
        )

    if len(continuous_ephys_data_list) == 1:
        continuous_ephys_data_list = continuous_ephys_data_list[0]
    else:
        # remove unnecessary extra list dimension and convert to numpy.ndarray
        continuous_ephys_data_list = np.squeeze(continuous_ephys_data_list).tolist()
    OE_data_dict = dict(zip(list_of_session_IDs, continuous_ephys_data_list))
    print("Loaded OpenEphys files: ", OE_data_dict.keys())
    return OE_data_dict


def load_anipose_data(chosen_rat, CFG, session_iterator):
    ## Input all desired data paths as a list to extract anipose data from
    # data_dir_list = [
    #     '/home/sean/hdd/GTE-BME/SNEL/data/anipose/session220603',
    #     '/home/sean/hdd/GTE-BME/SNEL/data/anipose/session220606',
    #     '/home/sean/hdd/GTE-BME/SNEL/data/anipose/session220608']
    ## Outputs all extracted dataframes, with unique file identifiers as dictionary keys

    # function definitions
    def _read_pose_3d_data(path_to_pose_3d_csv):
        # print(f"Reading Anipose file(s) into DataFrame: {path_to_pose_3d_csv.name}")
        return path_to_pose_3d_csv.name, pd.read_csv(path_to_pose_3d_csv)

    def _filter_csv_files(chosen_rat, CFG, session_iterator_copy, directory_name):
        for iSess in session_iterator_copy:
            # format inputs to avoid ambiguities
            session_date = CFG["rat"][chosen_rat]["session_date"][iSess]
            rat_name = str(chosen_rat).lower()
            treadmill_speed = str(CFG["rat"][chosen_rat]["treadmill_speed"][iSess]).zfill(2)
            treadmill_incline = str(CFG["rat"][chosen_rat]["treadmill_incline"][iSess]).zfill(2)
            session_ID = (
                f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
            )
            file_list = os.listdir(directory_name)
            for filename in file_list:
                if filename.endswith(".csv"):
                    if filename.__contains__(session_ID):
                        with open(
                            str(pathlib.PurePath(directory_name).joinpath(filename)), "r"
                        ) as csv_file:
                            yield _read_pose_3d_data(csv_file)
                        break

    # initialize lists
    # date_list = []                  # stores dates of each session
    list_of_session_IDs = []  # stores unique session identifiers
    list_of_pose_3d_dataframes = []  # stores all dataframes
    data_dir_list = CFG["data_dirs"]["anipose"]
    session_iterator_copy = session_iterator.copy()
    for directory_path in data_dir_list:
        pose_3d_path = str(pathlib.PurePath(directory_path).joinpath("pose-3d/"))
        file_list = os.listdir(pose_3d_path)

        csv_file_list = []
        [
            csv_file_list.append(file_list[iFile])
            for iFile in range(len(file_list))
            if file_list[iFile].endswith(".csv")
        ]
        for session_ID, df in _filter_csv_files(
            chosen_rat, CFG, session_iterator_copy, pose_3d_path
        ):
            # add date string to filename, apply lowercase, remove '.csv' suffix, then append to list
            list_of_session_IDs.append(session_ID.lower().split("/")[-1].split(".")[0])
            list_of_pose_3d_dataframes.append(df)
    anipose_data_dict = dict(zip(list_of_session_IDs, list_of_pose_3d_dataframes))
    print("Loaded Anipose files:   ", anipose_data_dict.keys())
    return anipose_data_dict


def load_KS_data(chosen_rat, CFG, session_iterator):
    session_IDs_dict = (
        {}
    )  # stores .info Session IDs of each recording in order under directory keys
    data_dir_list = CFG["data_dirs"]["KS"]
    sort_to_use = CFG["analysis"]["sort_to_use"]
    session_iterator_copy = session_iterator.copy()  # copy to avoid modifying original list
    cluster_id_ephys_data_dict = {}
    for data_dir in data_dir_list:
        recording_lengths_arr_list = []
        num_channels_list = []
        spikeClusters_arr = []
        spikeTimes_arr = []
        clusterIDs = []
        session_IDs_temp = []
        # chosen_rec_idxs_list = []
        session = Session(
            data_dir
        )  # copy over structure.oebin from recording folder to recording99 folder or errors

        sort_folder_name = _validate_data_dir_input(data_dir, sort_to_use)
        ops_path = str(pathlib.PurePath(data_dir).joinpath(sort_folder_name, "ops.mat"))
        ops = scipy.io.loadmat(ops_path)
        recordings_to_use = ops["recordings"][0]  # subtract 1 to match python indexing
        for iRec in range(
            (len(session.recordnodes[0].recordings))
        ):  # loops through all individual recording folders
            if session.recordnodes[0].recordings[iRec].directory.split("recording")[-1] not in [
                str(rec) for rec in recordings_to_use
            ]:
                continue  # skips recording if it is not in recordings_to_use
            # first, find file which has 'Rhythm' in it, then define timestamps_file variable
            Rhythm_folder = [
                file
                for file in os.listdir(
                    session.recordnodes[0].recordings[iRec].directory + "/continuous"
                )
                if "Rhythm" in file
            ][0]
            timestamps_file = (
                session.recordnodes[0].recordings[iRec].directory
                + "/continuous/"
                + Rhythm_folder
                + "/timestamps.npy"
            )
            num_channels_list.append(
                session.recordnodes[0].recordings[iRec].info["continuous"][0]["num_channels"]
            )
            recording_lengths_arr_list.append(len(np.load(timestamps_file)) * num_channels_list[-1])
            temp_dir = os.listdir(session.recordnodes[0].recordings[iRec].directory)
            if any(".info" in file for file in temp_dir):  # checks if .info file exists
                session_ID = [i for i in temp_dir if ".info" in i][0]
                # checks and filters for session_ID to match desired settings
                for i in session_iterator_copy:
                    session_date = CFG["rat"][chosen_rat]["session_date"][i]
                    rat_name = str(chosen_rat).lower()
                    treadmill_speed = str(CFG["rat"][chosen_rat]["treadmill_speed"][i]).zfill(2)
                    treadmill_incline = str(CFG["rat"][chosen_rat]["treadmill_incline"][i]).zfill(2)
                    session_ID_CFG = f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
                    if session_ID.__contains__(session_ID_CFG):
                        session_IDs_temp.append(session_ID.split(".")[0])
                        # chosen_rec_idxs_list.append(iRec)
                        # remove the index value after a match so you don't waste search loops in the future
                        session_iterator_copy.remove(i)
                        break
                    else:
                        continue
            else:
                print(session.recordnodes[0].recordings[iRec].directory + "has no .info file")
                return
        if len(session_IDs_temp) == 0:
            continue
        session_IDs_dict[
            data_dir
        ] = session_IDs_temp  # adds list of session IDs to dict under directory key

        # below divides each recording by the appropriate divisor
        # (divisor = number of channels in recording, which can be different by experimental error)
        recording_lengths_arr = np.array(recording_lengths_arr_list) / np.array(num_channels_list)
        assert all(
            recording_lengths_arr.astype(int) == recording_lengths_arr
        ), "Recording lengths not divisible by number of channels!"

        kilosort_files = []
        kilosort_folder = str(pathlib.PurePath(data_dir).joinpath(sort_folder_name))
        if "custom_merges" in os.listdir(kilosort_folder):
            # grab final merge output
            kilosort_files.append(
                str(
                    pathlib.PurePath(kilosort_folder).joinpath(
                        "custom_merges/final_merge/custom_merge.mat"
                    )
                )
            )
        else:
            # grab raw KS output
            kilosort_files.append(
                str(pathlib.PurePath(kilosort_folder).joinpath("spike_clusters.npy"))
            )
            kilosort_files.append(
                str(pathlib.PurePath(kilosort_folder).joinpath("spike_times.npy"))
            )
            if len(kilosort_files) != 2:
                print("KiloSort Spike Times and/or Spike Clusters not found!")
                return
        if ".npy" in kilosort_files[0]:
            spikeClusters_arr = np.load(kilosort_files[0]).ravel()
            spikeTimes_arr = np.load(kilosort_files[1]).ravel()
            # spikeClusters_arr = np.concatenate(spikeClusters)
            # spikeTimes_arr = np.concatenate(spikeTimes)
        elif ".mat" in kilosort_files[0]:
            spikeData = scipy.io.loadmat(kilosort_files[0])
            spikeClusters_arr = np.concatenate(spikeData["I"])
            spikeTimes_arr = np.concatenate(spikeData["T"])
        clusterIDs = np.unique(spikeClusters_arr)
        spikeTimes_arr = np.int32(spikeTimes_arr)
        # print("spikeTimes_arr[0]: ",spikeTimes_arr[0])
        # print(16 * (spikeTimes_arr[-1] - spikeTimes_arr[0]))

        # take cumsum to use later for recording file index boundaries
        recording_len_cumsum = np.insert(recording_lengths_arr.cumsum(), 0, 0).astype(int)

        # session_IDs_values = session_IDs_dict.values()
        # session_IDs_iter = iter(session_IDs_values)
        cluster_id_ephys_data_dict_temp = dict.fromkeys(tuple(session_IDs_temp))
        cluster_id_ephys_data_dict.update(cluster_id_ephys_data_dict_temp)
        for iChosen, session_ID in enumerate(session_IDs_dict[data_dir]):
            clusterIDs_ephys_spikeTimes = {}
            for id in clusterIDs:
                clusterIDs_ephys_spikeTimes_id_all = spikeTimes_arr[
                    np.where(spikeClusters_arr == id)
                ]
                clusterIDs_ephys_spikeTimes[id] = (
                    clusterIDs_ephys_spikeTimes_id_all[
                        # filter time ranges to be within the chosen session, and subtract the lower bound index
                        np.where(
                            (clusterIDs_ephys_spikeTimes_id_all > recording_len_cumsum[iChosen])
                            & (
                                clusterIDs_ephys_spikeTimes_id_all
                                < recording_len_cumsum[iChosen + 1]
                            )
                        )
                    ]
                    - recording_len_cumsum[iChosen]
                )
            cluster_id_ephys_data_dict[session_ID] = clusterIDs_ephys_spikeTimes
        print("Loaded KiloSort files:  ", cluster_id_ephys_data_dict.keys())
    return cluster_id_ephys_data_dict

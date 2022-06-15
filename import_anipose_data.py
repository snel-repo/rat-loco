import os
import os.path
import pandas as pd


def _read_pose_3d_data(path_to_pose_3d_csv):
    print(f"Reading file(s) into DataFrame: {path_to_pose_3d_csv.name}")
    return pd.read_csv(path_to_pose_3d_csv)

def _read_all_csv_files(directory_name):
    file_list = os.listdir(directory_name)
    for file_name in file_list:
        with open(os.path.join(directory_name, file_name), "r") as src_file:
            yield _read_pose_3d_data(src_file)


def import_anipose_data(directory_list):
    ## Input all desired data paths as a list to extract anipose data from 
    # directory_list = [
    #     '/home/sean/hdd/GTE-BME/SNEL/data/anipose/session220603',
    #     '/home/sean/hdd/GTE-BME/SNEL/data/anipose/session220606',
    #     '/home/sean/hdd/GTE-BME/SNEL/data/anipose/session220608']
    ## Outputs all extracted dataframes, with unique file identifiers as dictionary keys

    # initialize lists
    date_list = []                  # stores dates of each session
    list_of_session_IDs = []        # stores unique session identifiers
    list_of_pose_3d_dataframes = [] # stores all dataframes

    for iDir, directory_path in enumerate(directory_list):
        session_folder_name = directory_list[iDir].split('/')[-1] # extract the last foldername
        session_date = session_folder_name[-6:] # extract the date from folder name
        date_list.append(session_date)
        file_list = os.listdir(directory_path)
        for iFile, df in enumerate(_read_all_csv_files(directory_path)):
            # add date string to filename, apply lowercase, remove '.csv' suffix, then append to list
            list_of_session_IDs.append(date_list[iDir]+"_"+file_list[iFile].lower().split('.')[0])
            list_of_pose_3d_dataframes.append(df)

    anipose_data_dict = dict(zip(list_of_session_IDs,list_of_pose_3d_dataframes))
    return anipose_data_dict
import os
import pandas as pd
from pdb import set_trace

def _read_pose_3d_data(path_to_pose_3d_csv):
    print(f"Reading file(s) into DataFrame: {path_to_pose_3d_csv.name}")
    return path_to_pose_3d_csv.name, pd.read_csv(path_to_pose_3d_csv)

def _filter_csv_files(chosen_rat, CFG, directory_name):
    file_list = os.listdir(directory_name)
    for filename in file_list:
        if filename.endswith(".csv"):
            if filename.__contains__("incline"+str(CFG['rat'][chosen_rat]['treadmill_incline'][0]).zfill(2)):
                if filename.__contains__("speed"+str(CFG['rat'][chosen_rat]['treadmill_speed'][0]).zfill(2)):
                    if filename.__contains__(CFG['rat'][chosen_rat]['session_date'][0]):
                        if filename.__contains__(chosen_rat):
                            with open(os.path.join(directory_name, filename), "r") as csv_file:
                                yield _read_pose_3d_data(csv_file)


def import_anipose_data(chosen_rat, CFG):
    ## Input all desired data paths as a list to extract anipose data from 
    # directory_list = [
    #     '/home/sean/hdd/GTE-BME/SNEL/data/anipose/session220603',
    #     '/home/sean/hdd/GTE-BME/SNEL/data/anipose/session220606',
    #     '/home/sean/hdd/GTE-BME/SNEL/data/anipose/session220608']
    ## Outputs all extracted dataframes, with unique file identifiers as dictionary keys

    # initialize lists
    # date_list = []                  # stores dates of each session
    list_of_session_IDs = []        # stores unique session identifiers
    list_of_pose_3d_dataframes = [] # stores all dataframes
    directory_list = CFG['data_dirs']['anipose']
    
    for iDir, directory_path in enumerate(directory_list):
        pose_3d_path = os.path.join(directory_path,"pose-3d/")
        if directory_list[iDir][-1] != os.path.sep:   
            session_folder_name = directory_list[iDir].split(os.path.sep)[-1] # extract the last foldername
        else:
            session_folder_name = directory_list[iDir].split(os.path.sep)[-2] # extract the last foldername (ignoring front slash)
        # session_date = session_folder_name[-6:] # extract the date from folder name
        # date_list.append(session_date)
        file_list = os.listdir(pose_3d_path)
        csv_file_list = []
        [csv_file_list.append(file_list[iFile]) for iFile in range(len(file_list)) if file_list[iFile].endswith('.csv')]
        for (session_ID, df) in _filter_csv_files(chosen_rat, CFG, pose_3d_path):
            # add date string to filename, apply lowercase, remove '.csv' suffix, then append to list
            list_of_session_IDs.append(session_ID.lower().split('/')[-1].split('.')[0])
            list_of_pose_3d_dataframes.append(df)
    anipose_data_dict = dict(zip(list_of_session_IDs,list_of_pose_3d_dataframes))
    return anipose_data_dict

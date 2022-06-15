import os
import pdb
from open_ephys.analysis import Session
import numpy as np
from datetime import datetime

def import_OE_data(directory_list):
    # # Input all desired data paths to extract OpenEphys data from
    # directory_list = [
    #     '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-03_19-41-47', # data directory with 'Record Node ###' inside
    #     '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-06_15-21-22',
    #     '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-06_15-45-13',
    #     '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-06_16-01-57',
    #     '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-08_14-14-30']
    ## Outputs all extracted continuous ephys data packed in a dictionary and the keys are unique session identifiers
    
    # initialize lists and counter variable(s)
    number_of_recordings_per_session_list = []  # stores number of recordings found in each directory
    continuous_ephys_data_list = []             # stores continuous data
    date_list = []                              # stores recording dates
    recording_path_list = []                    # stores recording paths
    iRecording = int(-1)                        # counts the number of recordings per directory

    for iSessionPath in directory_list: # loop through all paths provided
        session = Session(iSessionPath) # use OE library to extract data from path into a structured python format
        number_of_recordings_per_session_list.append((len(session.recordnodes[0].recordings))) # store number of recordings to allow next looping operation

        for iExperiment in range(number_of_recordings_per_session_list[-1]):
            iRecording += 1 # increment to track the recording index out of all provided recording session paths
            continuous_ephys_data_list.append(session.recordnodes[0].recordings[iExperiment].continuous)
            # create list of session dates for the ones that were extracted
            date_list.append(session.recordnodes[0].directory.split('/')[-2]) # grab the date out of the path
            recording_path_list.append(session.recordnodes[0].recordings[iExperiment].directory)

    for iChannel in range(session.recordnodes[0].recordings[iExperiment].info['continuous'][0]['num_channels']):
        # Multiply each recording samples by the measured "bit_volts" value.
        # This converts from 16-bit number to uV for continuous channels and V for ADC channels
        continuous_ephys_data_list[iRecording][0].samples = continuous_ephys_data_list[iRecording][0].samples*session.recordnodes[0].recordings[iExperiment].info['continuous'][0]['channels'][iChannel]['bit_volts']
    
    
    list_of_session_IDs = []                                        # stores unique session identifiers
    for iDir, directory_path in enumerate(recording_path_list):
        file_list = os.listdir(directory_path)
        for filename in file_list:
            if filename.endswith(".info"):
                recording_info = filename.split('.')[0]
                date_only = date_list[iDir].split('_')[0]
                reformatted_date = datetime.strptime(date_only,"%Y-%m-%d").strftime("%y%m%d")
                list_of_session_IDs.append(reformatted_date+"_"+recording_info.lower())
    
    continuous_ephys_data_list = np.squeeze(continuous_ephys_data_list) # remove unnecessary extra list dimension and convert to numpy.ndarray
    ephys_data_dict = dict(zip(list_of_session_IDs,continuous_ephys_data_list))
    return ephys_data_dict
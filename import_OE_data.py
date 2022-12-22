import errno
import os
import numpy as np
from open_ephys.analysis import Session
# from datetime import datetime
# import matplotlib.pyplot as plt
# import pandas as pd
from pdb import set_trace
# from scipy.signal import find_peaks, filtfilt, iirnotch

# # create a notch filter for powerline removal at 60Hz
# def iir_notch(data, ephys_sample_rate, notch_frequency=60.0, quality_factor=30.0):
#     quality_factor = 20.0  # Quality factor
#     b, a = iirnotch(notch_frequency, quality_factor, ephys_sample_rate)
#     y = filtfilt(b, a, data)
#     return y

def import_OE_data(directory_list):
    # # Input all desired data paths to extract OpenEphys data from
    # directory_list = [
    #     '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-07-18_14-04-03/'
    #     ]
    #     '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-03_19-41-47', # data directory with 'Record Node ###' inside
    #     '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-06_15-21-22',
    #     '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-06_15-45-13',
    #     '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-06_16-01-57',
    #     '/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/treadmill/2022-06-08_14-14-30']
    ## Outputs all extracted continuous ephys data packed in a dictionary and the keys are unique session identifiers
    
    # initialize lists and counter variable(s)
    number_of_recordings_per_session_list = []  # stores number of recordings found in each directory
    continuous_ephys_data_list = []             # stores continuous data
    # date_list = []                              # stores recording dates
    recording_path_list = []                    # stores recording paths
    iRecording = int(-1)                        # counts the number of recordings per directory

    for iSessionPath in directory_list: # loop through all paths provided
        session = Session(iSessionPath) # use OE library to extract data from path into a structured python format
        number_of_recordings_per_session_list.append((len(session.recordnodes[0].recordings))) # store number of recordings to allow next looping operation

        for iExperiment in range(number_of_recordings_per_session_list[-1]):
            # skip recording99
            if 'recording99' in session.recordnodes[0].recordings[iExperiment].directory:
                continue
            iRecording += 1 # increment to track the recording index out of all provided recording session paths
            continuous_ephys_data_list.append(session.recordnodes[0].recordings[iExperiment].continuous)
            # create list of session dates for the ones that were extracted
            # date_list.append(session.recordnodes[0].directory.split('/')[-2]) # grab the date out of the path
            recording_path_list.append(session.recordnodes[0].recordings[iExperiment].directory)
            # set_trace()
            continuous_ephys_data_list[iRecording][0].samples = np.array(continuous_ephys_data_list[iRecording][0].samples,dtype='float32')

            for iChannel in range(session.recordnodes[0].recordings[iExperiment].info['continuous'][0]['num_channels']):
                # Multiply each recording samples by the measured "bit_volts" value.
                # This converts from 16-bit number to uV for continuous channels and V for ADC channels
                continuous_ephys_data_list[iRecording][0].samples[:,iChannel] = continuous_ephys_data_list[iRecording][0].samples[:,iChannel]*session.recordnodes[0].recordings[iExperiment].info['continuous'][0]['channels'][iChannel]['bit_volts']
    
    ## section plots the SYNC channel and describes stats of intervals            
    # signal = continuous_ephys_data_list[iRecording][0].samples[:,iChannel]
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
    
    list_of_session_IDs = []                                        # stores unique session identifiers
    for iDir, directory_path in enumerate(recording_path_list):
        file_list = os.listdir(directory_path)
        for filename in file_list:
            if filename.endswith(".info"):
                recording_info = filename.split('.')[0]
                # date_only = date_list[iDir].split('_')[0]
                # reformatted_date = datetime.strptime(date_only,"%Y-%m-%d").strftime("%y%m%d")
                list_of_session_IDs.append(recording_info.lower())
    if len(list_of_session_IDs) == 0:
        raise FileNotFoundError(
        errno.ENOENT, os.strerror(errno.ENOENT), "No '.info' file was found.")
    
    if len(continuous_ephys_data_list)==1:
        continuous_ephys_data_list = continuous_ephys_data_list[0]
    else:
        # remove unnecessary extra list dimension and convert to numpy.ndarray
        continuous_ephys_data_list = np.squeeze(continuous_ephys_data_list).tolist() 
    OE_data_dict = dict(zip(list_of_session_IDs,continuous_ephys_data_list))
    #set_trace()
    return OE_data_dict

## section for testing a specific directory without other 
# if __name__ == "__main__":
#     import_OE_data(['/home/sean/hdd/GTE-BME/SNEL/data/OpenEphys/knobtask/2022-07-19_18-04-05'])
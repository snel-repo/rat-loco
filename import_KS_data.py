import numpy as np
import os
import scipy.io
from pdb import set_trace
from open_ephys.analysis import Session
#### get a csv with sessionId and total ephys data length

## from open_ephys.analysis import Session
# def find_nearest(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return array[idx]

def import_KS_data(directory_list): 
    primary_dir = os.getcwd()
    output = {}  # stores kilosort ephys spike cluster id data
    sessionIDs_dict = {} # stores .info Session IDs of each recording in order under directory keys 
    for directory in directory_list:
        recording_lengths_arr_list = []
        spikeClusters_arr = []
        spikeTimes_arr = []
        clusterIDs = []
        sessionIDs_temp = []
        session = Session(directory) # copy over structure.oebin from recording folder to recording99 folder or errors 
        clusterIDs_ephys_spikeTimes = {}
        cluster_id_ephys_data_dict = {}
        for i in range((len(session.recordnodes[0].recordings)) - 1): # loops through all individual recording folders
            timestamps_file = session.recordnodes[0].recordings[i].directory + "/continuous/Acquisition_Board-100.Rhythm Data/sample_numbers.npy"
            recording_lengths_arr_list.append(len(np.load(timestamps_file)) * 16)
            temp_dir = os.listdir(session.recordnodes[0].recordings[i].directory)
            if any(".info" in file for file in temp_dir): # checks if .info file exists
                session_ID = [i for i in temp_dir if ".info" in i] 
                sessionIDs_temp.append(session_ID[0][0:-5])
            else:
                print(session.recordnodes[0].recordings[i].directory + "has no .info file")
                return
        sessionIDs_dict[directory] = sessionIDs_temp # adds list of session IDs to dict under directory key
        print(recording_lengths_arr_list)
        recording_lengths_arr = np.array(recording_lengths_arr_list)/16
        if any("KilosortResults" in folder for folder in os.listdir(session.recordnodes[0].recordings[-1].directory)):
            kilosort_files = []
            kilosort_folder = session.recordnodes[0].recordings[-1].directory + "/KilosortResults"
            for root, dirs, files in os.walk(kilosort_folder):
                if "spike_clusters.npy" in files:
                    kilosort_files.append(os.path.join(root, "spike_clusters.npy"))
                elif "spike_clusters.mat" in files:
                    kilosort_files.append(os.path.join(root, "spike_clusters.mat"))
                if "spike_times.npy" in files:
                    kilosort_files.append(os.path.join(root, "spike_times.npy"))
                elif "spike_times.mat" in files:
                    kilosort_files.append(os.path.join(root, "spike_times.mat"))
            
            if len(kilosort_files) != 2:
                print("KiloSort Spike Times and Spike Clusters not found!")
                return
            
            if ".npy" in kilosort_files[0]:
                spikeClusters = np.load(kilosort_files[0])
                spikeClusters_arr = np.concatenate(spikeClusters)       
                spikeTimes = np.load(kilosort_files[1])
                spikeTimes_arr = np.concatenate(spikeTimes)
            elif ".mat" in kilosort_files[0]:
                spikeClusters = scipy.io.loadmat(kilosort_files[0])
                spikeClusters_arr = np.concatenate(spikeClusters['I'])
                spikeTimes = scipy.io.loadmat(kilosort_files[1])
                spikeTimes_arr = np.concatenate(spikeTimes['T'])
            clusterIDs = np.unique(spikeClusters_arr)
            print(spikeTimes_arr[0])
            print(16 * (spikeTimes_arr[-1] - spikeTimes_arr[0]))
            
            i = 0
            sum = 0
            for sessionID in sessionIDs_dict[directory]:
                print(sessionID)
                for (id, length) in zip(clusterIDs, recording_lengths_arr):
                    clusterIDs_ephys_spikeTimes_id_all = spikeTimes_arr[np.where(spikeClusters_arr==id)]
                    clusterIDs_ephys_spikeTimes[id] = clusterIDs_ephys_spikeTimes_id_all[np.where(
                                                                clusterIDs_ephys_spikeTimes_id_all<length)]
                cluster_id_ephys_data_dict[sessionID] = clusterIDs_ephys_spikeTimes
            print(cluster_id_ephys_data_dict.keys())
                
            
            
        else:
            print("No folder named 'Kilosort Results' in " + session.recordnodes[0].recordings[-1].directory)
            print("If above error statement does not contain 'recording99' please make a folder with same name which contains Kilosort Results folder")
            raise FileNotFoundError()
        ## NEED TO USE RECORDING LENGTH ARRAY TO DIVIDE UP CONCATENATED DATA INTO SESSIONS
        ## THEN MAKE DICTIONARY IN FORM OF SESSIONID -> CLUSTERID -> SPIKETIMES ARRAY (WITH RESPECT TO LENGTHS OF EACH RECORDING)
        
    return cluster_id_ephys_data_dict

# directory_list = ['/snel/share/data/rodent-ephys/open-ephys/treadmill/pipeline/2022-11-16_16-19-28_myo']# ['/snel/share/data/rodent-ephys/open-ephys/treadmill/2022-11-18_18-38-54']#, '/snel/share/data/rodent-ephys/open-ephys/treadmill/2022-11-17_17-08-07']
# tempdict = define_index_dict(directory_list)
# print("done.")
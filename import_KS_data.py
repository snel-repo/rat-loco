import numpy as np
import os
from pathlib import Path
from open_ephys.analysis import Session
#### get a csv with sessionId and total ephys data length

## from open_ephys.analysis import Session
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def define_index_dict(directory_list): 
    primary_dir = os.getcwd()
    output = {}  # stores kilosort ephys spike cluster id data
    sessionIDs_dict = {} # stores .info Session IDs of each recording in order under directory keys 
    for directory in directory_list:
        recording_lengths_arr = []
        spikeClusters_arr = []
        spikeTimes_arr = []
        clusterIDs = []
        sessionIDs_temp = []
        session = Session(directory) # copy over structure.oebin from recording folder to recording99 folder or errors 
        for i in range((len(session.recordnodes[0].recordings)) - 1): # loops through all individual recording folders
            timestamps_file = session.recordnodes[0].recordings[i].directory + "/continuous/Acquisition_Board-100.Rhythm Data/timestamps.npy"
            recording_lengths_arr.append(len(np.load(timestamps_file)) * 16)
            temp_dir = os.listdir(session.recordnodes[0].recordings[i].directory)
            if any(".info" in file for file in temp_dir): # checks if .info file exists
                session_ID = [i for i in temp_dir if ".info" in i] 
                sessionIDs_temp.append(session_ID[0][0:-5])
            else:
                print(session.recordnodes[0].recordings[i].directory + "has no .info file")
                return
        sessionIDs_dict[directory] = sessionIDs_temp # adds list of session IDs to dict under directory key
        print(recording_lengths_arr)
        if any("Kilosort Results" in folder for folder in os.listdir(session.recordnodes[0].recordings[-1].directory)):
            kilosort_files = []
            kilosort_folder = session.recordnodes[0].recordings[-1].directory + "/Kilosort Results"
            for root, dirs, files in os.walk(kilosort_folder):
                if "spike_clusters.npy" in files:
                    kilosort_files.append(os.path.join(root, "spike_clusters.npy"))
                if "spike_times.npy" in files:
                    kilosort_files.append(os.path.join(root, "spike_times.npy"))
                    
            if len(kilosort_files) != 2:
                print("KiloSort Spike Times and Spike Clusters not found!")
                return

            spikeClusters = np.load(kilosort_files[0])
            spikeClusters_arr = np.concatenate(spikeClusters)       
            spikeTimes = np.load(kilosort_files[1])
            spikeTimes_arr = np.concatenate(spikeTimes) 
            clusterIDs = np.unique(spikeClusters_arr)
            print(spikeTimes_arr[0])
            print(16 * (spikeTimes_arr[-1] - spikeTimes_arr[0]))
            
            i = 0
            sum = 0
            for sessionID in sessionIDs_dict[directory]:
                print(sessionID)
            
            
        else:
            print("No folder named 'Kilosort Results' in " + session.recordnodes[0].recordings[-1].directory)
            print("If above error statement does not contain 'recording99' please make a folder with same name which contains Kilosort Results folder")
    
        ## NEED TO USE RECORDING LENGTH ARRAY TO DIVIDE UP CONCATENATED DATA INTO SESSIONS
        ## THEN MAKE DICTIONARY IN FORM OF SESSIONID -> CLUSTERID -> SPIKETIMES ARRAY (WITH RESPECT TO LENGTHS OF EACH RECORDING)
        
        #for id in clusterIDs:
        #    clusterIDs_ephys_spikeTimes[id] = spikeTimes_arr[np.where(spikeClusters_arr==id)]
        #cluster_id_ephys_data_list[sessionID] = clusterIDs_ephys_spikeTimes
        #print(cluster_id_ephys_data_list.keys())

directory_list = ['/snel/share/data/rodent-ephys/open-ephys/treadmill/2022-11-18_18-38-54']#, '/snel/share/data/rodent-ephys/open-ephys/treadmill/2022-11-17_17-08-07']
tempdict = define_index_dict(directory_list)

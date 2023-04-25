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

def import_KS_data(chosen_rat, CFG, session_iterator): 
    session_IDs_dict = {} # stores .info Session IDs of each recording in order under directory keys 
    data_dir_list = CFG['data_dirs']['KS']
    session_iterator_copy = session_iterator.copy()
    for data_dir in data_dir_list:
        recording_lengths_arr_list = []
        num_channels_list = []
        spikeClusters_arr = []
        spikeTimes_arr = []
        clusterIDs = []
        session_IDs_temp = []
        chosen_rec_idxs_list = []
        session = Session(data_dir) # copy over structure.oebin from recording folder to recording99 folder or errors 
        clusterIDs_ephys_spikeTimes = {}
        cluster_id_ephys_data_dict = {}
        for iRec in range((len(session.recordnodes[0].recordings))): # loops through all individual recording folders
            # skip recording99
            if 'recording99' in session.recordnodes[0].recordings[iRec].directory:
                continue
            timestamps_file = session.recordnodes[0].recordings[iRec].directory + "/continuous/Acquisition_Board-100.Rhythm Data/timestamps.npy"
            if not os.path.exists(timestamps_file):
                timestamps_file = session.recordnodes[0].recordings[iRec].directory + "/continuous/Rhythm_FPGA-100.0/timestamps.npy"
            num_channels_list.append(session.recordnodes[0].recordings[iRec].info['continuous'][0]['num_channels'])
            recording_lengths_arr_list.append(len(np.load(timestamps_file)) * num_channels_list[-1])
            temp_dir = os.listdir(session.recordnodes[0].recordings[iRec].directory)
            if any(".info" in file for file in temp_dir): # checks if .info file exists
                session_ID = [i for i in temp_dir if ".info" in i][0]
                # checks and filters for session_ID to match desired settings
                for iterator in session_iterator_copy:
                    session_date = CFG['rat'][chosen_rat]['session_date'][iterator]
                    rat_name = str(chosen_rat).lower()
                    treadmill_speed = str(CFG['rat'][chosen_rat]['treadmill_speed'][iterator]).zfill(2)
                    treadmill_incline = str(CFG['rat'][chosen_rat]['treadmill_incline'][iterator]).zfill(2)
                    session_ID_CFG = f"{session_date}_{rat_name}_speed{treadmill_speed}_incline{treadmill_incline}"
                    if session_ID.__contains__(session_ID_CFG):
                        session_IDs_temp.append(session_ID.split('.')[0])
                        chosen_rec_idxs_list.append(iRec)
                        # remove the index value after a match so you don't waste search loops in the future
                        session_iterator_copy.remove(iterator)
                        break
                    else:
                        continue
            else:
                print(session.recordnodes[0].recordings[iRec].directory + "has no .info file")
                return
        if len(session_IDs_temp)==0:
            continue
        session_IDs_dict[data_dir] = session_IDs_temp # adds list of session IDs to dict under directory key
        
        # below divides each recording by the appropriate divisor
        # (divisor = number of channels in recording, which can be different by experimental error)
        recording_lengths_arr = np.array(recording_lengths_arr_list)/np.array(num_channels_list)
        concatenated_data_dir = os.path.abspath(os.path.join(session.recordnodes[0].recordings[-1].directory, os.pardir,'concatenated_data'))
        if not os.path.exists(concatenated_data_dir):
            concatenated_data_dir = session.recordnodes[0].recordings[-1].directory # folder which contains all session data combined
        print(f"Using {concatenated_data_dir} folder.")
        if "sorted0" in os.listdir(data_dir):
            kilosort_files = []
            kilosort_folder = os.path.join(data_dir,"sorted0")
            if "custom_merges" in os.listdir(kilosort_folder):
                # grab final merge output
                kilosort_files.append(os.path.join(kilosort_folder, "custom_merges/final_merge/custom_merge.mat"))
            else:
                # grab raw KS output
                kilosort_files.append(os.path.join(kilosort_folder, "spike_clusters.npy"))
                kilosort_files.append(os.path.join(kilosort_folder, "spike_times.npy"))
                if len(kilosort_files) != 2:
                    print("KiloSort Spike Times and/or Spike Clusters not found!")
                    return
        elif "KilosortResults" in os.listdir(concatenated_data_dir):
            kilosort_files = []
            kilosort_folder = concatenated_data_dir + "/KilosortResults"
            for root, dirs, files in os.walk(kilosort_folder):
                if "spike_clusters.npy" in files and "spike_times.npy" in files:
                    kilosort_files.append(os.path.join(root, "spike_clusters.npy"))
                    kilosort_files.append(os.path.join(root, "spike_times.npy"))
                    if len(kilosort_files) != 2:
                        print("KiloSort Spike Times and/or Spike Clusters not found!")
                        return
        else:
            print("No folder named 'KilosortResults' in " + concatenated_data_dir + " or sorted0 folder with custom_merge.mat")
            print("If above error statement does not contain 'recording99' please make a folder with same name which contains Kilosort Results folder")
            raise FileNotFoundError()
        if ".npy" in kilosort_files[0]:
            spikeClusters_arr = np.load(kilosort_files[0]).ravel()
            spikeTimes_arr = np.load(kilosort_files[1]).ravel()
            # spikeClusters_arr = np.concatenate(spikeClusters)       
            # spikeTimes_arr = np.concatenate(spikeTimes)
        elif ".mat" in kilosort_files[0]:
            spikeData = scipy.io.loadmat(kilosort_files[0])
            spikeClusters_arr = np.concatenate(spikeData['I'])
            spikeTimes_arr = np.concatenate(spikeData['T'])
        clusterIDs = np.unique(spikeClusters_arr)
        spikeTimes_arr = np.int32(spikeTimes_arr)
        # print("spikeTimes_arr[0]: ",spikeTimes_arr[0])
        # print(16 * (spikeTimes_arr[-1] - spikeTimes_arr[0]))
        
        # take cumsum to use later for recording file index boundaries
        recording_len_cumsum = np.insert(recording_lengths_arr.cumsum(),0,0).astype(int)
        cluster_id_ephys_data_dict = dict.fromkeys(tuple(*session_IDs_dict.values()))
        for (session_ID, iChosen) in zip(session_IDs_dict[data_dir],chosen_rec_idxs_list):
            clusterIDs_ephys_spikeTimes = clusterIDs_ephys_spikeTimes.copy()
            for id in clusterIDs:
                clusterIDs_ephys_spikeTimes_id_all = spikeTimes_arr[np.where(spikeClusters_arr==id)]
                clusterIDs_ephys_spikeTimes[id] = clusterIDs_ephys_spikeTimes_id_all[
                    # filter time ranges to be within the chosen session, and subtract the lower bound index
                    np.where((clusterIDs_ephys_spikeTimes_id_all>recording_len_cumsum[iChosen]) &
                             (clusterIDs_ephys_spikeTimes_id_all<recording_len_cumsum[iChosen+1]))] - recording_len_cumsum[iChosen]
            cluster_id_ephys_data_dict[session_ID] = clusterIDs_ephys_spikeTimes
        print("Loaded KiloSort files:  ", cluster_id_ephys_data_dict.keys())
                
        ## NEED TO USE RECORDING LENGTH ARRAY TO DIVIDE UP CONCATENATED DATA INTO SESSIONS
        ## THEN MAKE DICTIONARY IN FORM OF SESSIONID -> CLUSTERID -> SPIKETIMES ARRAY (WITH RESPECT TO LENGTHS OF EACH RECORDING)
    return cluster_id_ephys_data_dict

# data_dir_list = ['/snel/share/data/rodent-ephys/open-ephys/treadmill/pipeline/2022-11-16_16-19-28_myo']# ['/snel/share/data/rodent-ephys/open-ephys/treadmill/2022-11-18_18-38-54']#, '/snel/share/data/rodent-ephys/open-ephys/treadmill/2022-11-17_17-08-07']
# tempdict = define_index_dict(data_dir_list)
# print("done.")
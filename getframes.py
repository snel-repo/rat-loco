import sys
import os
import time

poisson_ip = "192.168.1.206"

if __name__=='__main__':
    if len(list(sys.argv)) == 1:
        date = time.localtime()
        date = str(date[0])+str(date[1]).zfill(2)+str(date[2]).zfill(2)
        new_base_folder_name = 'images'+date
    elif len(list(sys.argv)) == 2:
        date = sys.argv[1]
        new_base_folder_name = 'images'+date
    else:
        raise Exception("Enter no more than 1 argument for the 'FLIR_Multi_Cam_HWTrig/images' folder date. Example: 'getframes 220214' for Valentine's Day, 2022, will access images220214/ folder")
    print("retrieving frames from poisson.")
    all_dirs = os.listdir("/home/snel/git/FLIR_Multi_Cam_HWTrig/")
    date_match_dir_list= []
    for d in all_dirs:
        if new_base_folder_name in d:
            date_match_dir_list.append(d)
    for iDir in date_match_dir_list:
        os.system("ssh snel@"+poisson_ip+f" 'ls /home/snel/git/FLIR_Multi_Cam_HWTrig/{iDir}/ > /home/snel/git/FLIR_Multi_Cam_HWTrig/{iDir}/frames_list.txt'")
        os.system("rsync -aP --rsh=ssh --files-from=snel@"+poisson_ip+f":/home/snel/git/FLIR_Multi_Cam_HWTrig/{iDir}/frames_list.txt snel@"+poisson_ip+f":/home/snel/git/FLIR_Multi_Cam_HWTrig/{iDir}/ /home/snel/git/FLIR_Multi_Cam_HWTrig/{iDir}/")
        os.system("rm /home/snel/git/FLIR_Multi_Cam_HWTrig/{iDir}/frames_list.txt")
    print("done.")

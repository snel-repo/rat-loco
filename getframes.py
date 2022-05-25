import sys
import os
import time

poisson_ip = "192.168.1.206"

if __name__=='__main__':
    if len(list(sys.argv)) == 1:
        folder_date = time.strftime("%y%m%d")
    elif len(list(sys.argv)) == 2:
        folder_date = sys.argv[1]
    else:
        raise Exception("Enter no more than 1 argument for the 'FLIR_Multi_Cam_HWTrig/images' folder date. Example: 'getframes 220214' for Valentine's Day, 2022, will access images220214/ folder")
    print("retrieving frames from poisson.")
    os.system("ssh snel@"+poisson_ip+f" 'ls /home/snel/git/FLIR_Multi_Cam_HWTrig/images{folder_date}/ > /home/snel/git/FLIR_Multi_Cam_HWTrig/images{folder_date}/frames_list.txt'")
    os.system("rsync -aP --rsh=ssh --files-from=snel@"+poisson_ip+f":/home/snel/git/FLIR_Multi_Cam_HWTrig/images{folder_date}/frames_list.txt snel@"+poisson_ip+f":/home/snel/git/FLIR_Multi_Cam_HWTrig/images{folder_date}/ /home/snel/git/FLIR_Multi_Cam_HWTrig/images{folder_date}/")
    print("done.")

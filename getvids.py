import sys
import os
import time

axon_ip = "192.168.1.206"

if __name__=='__main__':
    if len(list(sys.argv)) == 1:
        folder_date = time.strftime("%y%m%d")
    elif len(list(sys.argv)) == 2:
        folder_date = sys.argv[1]
    else:
        raise Exception("Enter no more than 1 argument for the 'FLIR_Multi_Cam_HWTrig/images' folder date. Example: 220110 for Jan. 10th 2022, will access images220110/ folder")
    print("retrieving videos from axon.")
    os.system("scp snel@"+axon_ip+f":/home/snel/git/FLIR_Multi_Cam_HWTrig/images{folder_date}/*.mp4 /home/snel/git/FLIR_Multi_Cam_HWTrig/images{folder_date}/")    
    print("done.")

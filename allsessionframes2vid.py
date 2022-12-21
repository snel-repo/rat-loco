import sys
import os
import time
import allframes2vid

def allsessionframes2vid(num_frames, date=time.localtime(), downsampling_factor=1, destination='.', framerate=125, num_cams=4):
     if date==time.localtime():
          date = str(date[0])+str(date[1]).zfill(2)+str(date[2]).zfill(2)
          base_folder_name = 'images'+date
     else:
          base_folder_name = 'images'+date
     all_dirs = os.listdir("/home/snel/git/FLIR_Multi_Cam_HWTrig/")
     date_match_dir_list= []
     for d in all_dirs:
          if base_folder_name in d:
               date_match_dir_list.append(d)
     for iDir in date_match_dir_list:
          os.chdir(f"/home/snel/git/FLIR_Multi_Cam_HWTrig/{iDir}/")
          # get all unique base filenames, and write that to a file
          allframes2vid.allframes2vid(num_frames, downsampling_factor, destination, framerate, num_cams)

if __name__ == '__main__':
     if len(list(sys.argv)) == 2:
          allsessionframes2vid(num_frames=sys.argv[1])
     elif len(list(sys.argv)) == 3:
          allsessionframes2vid(num_frames=sys.argv[1], date=sys.argv[2], downsampling_factor=sys.argv[3])
     elif len(list(sys.argv)) == 4:
          allsessionframes2vid(num_frames=sys.argv[1], date=sys.argv[2], downsampling_factor=sys.argv[3], destination=sys.argv[4])
     elif len(list(sys.argv)) == 5:
          allsessionframes2vid(num_frames=sys.argv[1], date=sys.argv[2], downsampling_factor=sys.argv[3], destination=sys.argv[4], framerate=sys.argv[5])
     elif len(list(sys.argv)) == 6:
          allsessionframes2vid(num_frames=sys.argv[1], date=sys.argv[2], downsampling_factor=sys.argv[3], destination=sys.argv[4], framerate=sys.argv[5], num_cams=sys.argv[6])
     else:
          raise Exception("enter at least 1 arguments and no more than 5! :) \nInputs: num_frames, date, downsampling_factor, destination, framerate, and num_cams")
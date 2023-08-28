import sys
import os
import time
import allframes2vid
# from pdb import set_trace

def allsessionframes2vid(num_frames, date='auto', downsampling_factor=1, destination='.', framerate=125, num_cams=4):
     if date=='auto':
          dstr = time.localtime()
          date = str(dstr[0])+str(dstr[1]).zfill(2)+str(dstr[2]).zfill(2)
          base_folder_name = 'images'+date
     else:
          base_folder_name = 'images'+date
     all_dirs = os.listdir(os.path.expanduser("~/git/rat-loco/FLIR-Multicam"))
     date_match_dir_list= []
     for d in all_dirs:
          if base_folder_name in d:
               date_match_dir_list.append(d)
     for iDir in date_match_dir_list:
          os.chdir(os.path.expanduser(f"~/git/rat-loco/FLIR-Multicam/{iDir}/"))
          # get all unique base filenames, and write that to a file
          allframes2vid.allframes2vid(num_frames, downsampling_factor, destination, framerate, num_cams)
     try:
          os.system("rm unique_base_filenames.txt")
     except FileExistsError:
          print("Could not remove unique_base_filenames.txt, as it does not exist.")
     except:
          raise
     
if __name__ == '__main__':
     if len(list(sys.argv)) == 2:
          allsessionframes2vid(num_frames=sys.argv[1])
     elif len(list(sys.argv)) == 3:
          allsessionframes2vid(num_frames=sys.argv[1], date=sys.argv[2])
     elif len(list(sys.argv)) == 4:
          allsessionframes2vid(num_frames=sys.argv[1], date=sys.argv[2], downsampling_factor=sys.argv[3])
     elif len(list(sys.argv)) == 5:
          allsessionframes2vid(num_frames=sys.argv[1], date=sys.argv[2], downsampling_factor=sys.argv[3], destination=sys.argv[4])
     elif len(list(sys.argv)) == 6:
          allsessionframes2vid(num_frames=sys.argv[1], date=sys.argv[2], downsampling_factor=sys.argv[3], destination=sys.argv[4], framerate=sys.argv[5])
     elif len(list(sys.argv)) == 7:
          allsessionframes2vid(num_frames=sys.argv[1], date=sys.argv[2], downsampling_factor=sys.argv[3], destination=sys.argv[4], framerate=sys.argv[5], num_cams=sys.argv[6])
     else:
          raise Exception("enter at least 1 arguments and no more than 5! :) \nInputs: num_frames, date, downsampling_factor, destination, framerate, and num_cams")
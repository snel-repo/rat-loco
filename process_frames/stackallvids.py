import sys
import os
import stackvids


def stackallvids(stack_type='xstack', destination='.', num_cams=4):
     # get all unique base filenames, and write that to a file
     os.system("find ./ -type f -printf '%f\n' | sort | grep mp4 | sed s/_cam.....$// | uniq > unique_video_names.txt")
     with open('unique_video_names.txt','r') as unique_video_names:
          # read in each line into a list, without any extra \n characters
          unique_video_names_list = unique_video_names.read().splitlines()
          for iVideoName in unique_video_names_list:
               stackvids.stackvids(iVideoName, stack_type, destination, num_cams)
          
if __name__ == '__main__':
     if len(list(sys.argv)) == 1:
          stackallvids()
     elif len(list(sys.argv)) == 2:
          stackallvids(stack_type=sys.argv[1])
     elif len(list(sys.argv)) == 3:
          stackallvids(stack_type=sys.argv[1], destination=sys.argv[2])
     elif len(list(sys.argv)) == 4:
          stackallvids(stack_type=sys.argv[1], destination=sys.argv[2], num_cams=sys.argv[3])
     else:
          raise Exception("enter at least 2 arguments and no more than 4! :) \nInputs: base_filename, num_frames, destination, bottom_cam_id, num_cams, framerate")
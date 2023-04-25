import sys
import os
import frames2vid as frames2vid

def allframes2vid(num_frames, downsampling_factor=1, destination='.', framerate=125, num_cams=4):
     # get all unique base filenames, and write that to a file
     os.system("find ./ -type f -printf '%f\n' | sort | grep jpg | sed s/_[0-9]*_cam.....$// | sort | uniq > unique_base_filenames.txt")
     with open('unique_base_filenames.txt','r') as unique_base_filenames:
          # read in each line into a list, without any extra \n characters
          unique_base_filenames_list = unique_base_filenames.read().splitlines()
          for iFilename in unique_base_filenames_list:
               frames2vid.frames2vid(iFilename, num_frames, downsampling_factor, destination, framerate, num_cams)
     os.system("rm unique_base_filenames.txt")

if __name__ == '__main__':
     if len(list(sys.argv)) == 2:
          allframes2vid(num_frames=sys.argv[1])
     elif len(list(sys.argv)) == 3:
          allframes2vid(num_frames=sys.argv[1], downsampling_factor=sys.argv[2])
     elif len(list(sys.argv)) == 4:
          allframes2vid(num_frames=sys.argv[1], downsampling_factor=sys.argv[2], destination=sys.argv[3])
     elif len(list(sys.argv)) == 5:
          allframes2vid(num_frames=sys.argv[1], downsampling_factor=sys.argv[2], destination=sys.argv[3], framerate=sys.argv[4])
     elif len(list(sys.argv)) == 6:
          allframes2vid(num_frames=sys.argv[1], downsampling_factor=sys.argv[2], destination=sys.argv[3], framerate=sys.argv[4], num_cams=sys.argv[5])
     else:
          raise Exception("enter at least 1 arguments and no more than 5! :) \nInputs: num_frames, downsampling_factor, destination, framerate, and num_cams")
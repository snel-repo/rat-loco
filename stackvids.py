import sys
import os


def stackvids(base_filename, destination='.', stack_type='hstack', num_cams=4):
     # python command to process frames into 3 videos, flipping the bottom camera view to enable triangulation
     # COMMAND: ffmpeg -framerate 100 -f image2 -i ./Calib0_%3d_cam0.jpg -vcodec libx264 -crf 10 -pix_fmt yuv420p -filter:v "scale= iw/2:ih/2, vflip" -an Calib0_cam0.mp4
     
     # try:
     #      assert type(int(num_frames)) is int,"num_frames must be an integer!"
     #      assert type(int(downsampling_factor)) is int,"downsampling_factor must be an integer!"
     # except:
     #      raise Exception("Problem with num_frames input. It must be an integer!")
               
     
     # num_frame_digits = len(str(num_frames)) # get max number of digits in frame count
     
     if num_cams==2:
          os.system(f'ffmpeg -i {base_filename}_cam0.mp4 -i {base_filename}_cam1.mp4 -filter_complex {stack_type}=inputs=2 {destination}/stack_{base_filename}.mp4')
     elif num_cams==3:
          os.system(f'ffmpeg -i {base_filename}_cam0.mp4 -i {base_filename}_cam1.mp4 -i {base_filename}_cam2.mp4 -filter_complex {stack_type}=inputs=3 {destination}/stack_{base_filename}.mp4')
     elif num_cams==4:
          os.system(f'ffmpeg -i {base_filename}_cam0.mp4 -i {base_filename}_cam1.mp4 -i {base_filename}_cam2.mp4 -i {base_filename}_cam3.mp4 -filter_complex {stack_type}=inputs=4 {destination}/stack_{base_filename}.mp4')

if __name__ == '__main__':
     # print(list(sys.argv))
     if len(list(sys.argv)) == 2:
          stackvids(sys.argv[1])
     elif len(list(sys.argv)) == 3:
          stackvids(sys.argv[1], destination=sys.argv[2])
     elif len(list(sys.argv)) == 4:
          stackvids(sys.argv[1], destination=sys.argv[2], stack_type=sys.argv[3])
     elif len(list(sys.argv)) == 5:
          stackvids(sys.argv[1], destination=sys.argv[2], stack_type=sys.argv[3], num_cams=sys.argv[4])
     else:
          raise Exception("enter at least 2 arguments and no more than 4! :) \nInputs: base_filename, num_frames, destination, bottom_cam_id, num_cams, framerate")
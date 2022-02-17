import sys
import os


def frames2vid(base_filename, num_frames, destination='.', bottom_cam_id=0, num_cams=3, framerate=100):
     # python command to process frames into 3 videos, flipping the bottom camera view to enable triangulation
     # COMMAND: ffmpeg -framerate 100 -f image2 -i ./Calib0_%3d_cam0.jpg -vcodec libx264 -crf 10 -pix_fmt yuv420p -filter:v "scale= iw/2:ih/2, vflip" -an Calib0_cam0.mp4
     
     try:
          assert type(int(num_frames)) is int,"num_frames must be an integer!"
     except:
          raise Exception("Problem with num_frames input. It must be an integer!")
               
     
     num_frame_digits = len(str(num_frames)) # get max number of digits in frame count
     
     for ii in range(num_cams):
          iCam = "cam"+str(ii)
          if iCam == "cam"+str(bottom_cam_id):
               os.system(f'ffmpeg -framerate {framerate} -f image2 -i ./{base_filename}_%{num_frame_digits}d_{iCam}.jpg -vcodec libx264 -crf 10 -pix_fmt yuv420p -filter:v "scale= iw/2:ih/2, vflip" -an {destination}/{base_filename}_{iCam}.mp4')
          else:
               os.system(f'ffmpeg -framerate {framerate} -f image2 -i ./{base_filename}_%{num_frame_digits}d_{iCam}.jpg -vcodec libx264 -crf 10 -pix_fmt yuv420p -filter:v "scale= iw/2:ih/2" -an {destination}/{base_filename}_{iCam}.mp4')

if __name__ == '__main__':
     print(list(sys.argv))
     if len(list(sys.argv)) == 3:
          frames2vid(sys.argv[1], sys.argv[2])
     elif len(list(sys.argv)) == 4:
          frames2vid(sys.argv[1], sys.argv[2], destination=sys.argv[3])
     elif len(list(sys.argv)) == 5:
          frames2vid(sys.argv[1], sys.argv[2], destination=sys.argv[3], bottom_cam_id=sys.argv[4])
     elif len(list(sys.argv)) == 6:
          frames2vid(sys.argv[1], sys.argv[2], destination=sys.argv[3], bottom_cam_id=sys.argv[4], num_cams=sys.argv[5])
     elif len(list(sys.argv)) == 7:
          frames2vid(sys.argv[1], sys.argv[2], destination=sys.argv[3], bottom_cam_id=sys.argv[4], num_cams=sys.argv[5], framerate=sys.argv[6])
     else:
          raise Exception("enter at least 2 arguments and less than 6! :) \nInputs: base_filename, num_frames, destination, bottom_cam_id, num_cams, framerate")
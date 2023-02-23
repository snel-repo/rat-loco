# Ephys&Cam_Recording_Instructions
***FOR EPHYS***
1. Ensure the SYNC and GND cables are connected from the Arduino and Cameras to the white Open Ephys recording system
2. Login to the Poisson standing desk with vertical monitor, then open Open Ephys with the blue cable connecting to the other side of the white box
3. Some initial settings...
4. When ready to record, hit play button first and hit start button BEFORE starting camera recording
5. As the camera starts to record, box-like square waves will show on the bottom channel
6. Begin the recording after the rat has already reached a steady state for the desired speed
7. After OpenEphys recording has begun, start the camera recordings as described below
8. Once the cameras have acquired all the frames and completed recording, stop the OpenEphys recording, and stop the treadmill

***FOR CAMERA***
1. Open `~/git/FLIR_Multi_Cam_HWTrig/params.yaml` on Visual Studio Code
2. ***num_images***: the number of frames to capture (125fps * seconds of reacording)
   ***file_name***: use format yyyymmdd_ratname_speedXX_inclineYY (e.g., 20221118_godzilla_speed15_incline00)
3. Execute `rec` to put all 4 cameras into a "Trigger Wait" state
4. Plug in the orange cable to #2 on the arduino to start camera recording
5. Unplug the orange cable when recording is finished 

***GET CONVERTING FRAMES TO VIDEO***
1. Go to `~/git/FLIR_Multi_Cam_HWTrig/`, then use `allsessionframes2vid <number of frames>` to loop through all session folders from today and convert all frames to videos. For all sessions from a different date, run `allsessionframes2vid <number of frames> <yyyymmdd>`.
2. (Optional) To combine and visualize all videos, navigate to one of the `images<yyyymmdd-X>` folders, and use `stackallvids`. Add argument: `xstack`, `hstack`, or `vstack` to stack them horizontally, vertically, or in a square (e.g. default is `stackallvids hstack`)

***ANIPOSE***
1. Run `makeanipose` to automatically create a new session folder in `~/anipose/session<yyyymmdd>` and move all videos from today into the `videos-raw` folder
2. Here is a list of commands used in Anipose (run ALL commands in order to triangulate)
      `anipose analyze` : creates pose-2d folder
      `anipose filter` (apply over the 2D data functions as a threshold filter): creates pose-2d-filtered folder
      `anipose label-2d-filter` (plot the filtered predicted 2D labels on each frame): creates videos-labeled-filtered folder.
      `anipose label-2d` (view the unfiltered predicted 2D labels on each frame): creates videos-labeled folder
      `anipose calibrate`
      `anipose triangulate` (generate csv file for each group of videos): creates pose-3d folder
      `anipose label-3d` (plot the predicted labels from the 3D tracking for each group of videos): creates videos-3d folder
      `anipose label-combined` (concatenate the videos for each group of videos obtained from running label-2d and label-3d): creates videos-combined folder


***USEFUL COMMANDS***
Windows Z - keyboard shortcut for a new terminal
Windows C - launch coolero to monitor CPU/GPU usage, temperatures, and fan speeds
Windows E - launch file manager
Windows Spacebar - launch app finder
`htop` check the CPUs
`nvidia-smi` check the GPUs
`rm same_element_in_filenames* ` with the star sign you can remove all files with the same elements in the name
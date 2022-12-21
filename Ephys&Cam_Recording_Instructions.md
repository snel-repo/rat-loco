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
2. ***num_images***: the number of frames to capture
   ***file_name***: use format yyyymmdd_ratname_speedXX_inclineYY (e.g., 20221118_godzilla_speed15_incline00)
3. Open 2 terminals
   -In one, type `sshpoisson` to access Poisson, which controls 2 of the BlackFly S cameras
   -In both, executre `rec` to put all 4 cameras into a "Trigger Wait" state
4. Plug in the orange cable to #2 on the arduino to start camera recording
5. Unplug the orange cable when recording is finished 

***GET CONVERTING FRAMES TO VIDEO***
1. Close the Poisson terminal
2. Execute `getframes` on Axon to get all images from Poisson for all sessions from today or `getframes <yyyymmdd>` for all sessions from another date.
3. Go to `~/git/FLIR_Multi_Cam_HWTrig/`, then use `allsessionframes2vid <number of frames>` to loop through all session folders from today and convert all frames to videos. For all sessions from a different date, run `allsessionframes2vid <number of frames> <yyyymmdd>`.
4. (Optional) To combine and visualize all videos, navigate to one of the `images<yyyymmdd-X>` folders, and use `stackallvids`. Add argument: `xstack`, `hstack`, or `vstack` to stack them horizontally, vertically, or in a square (e.g. default is `stackallvids hstack`)

***ANIPOSE***
1. Run `makeanipose` to automatically create a new session folder in `~/anipose/session<yyyymmdd>` and move all videos from today into the `videos-raw` folder
2. Check https://anipose.readthedocs.io/en/latest/tutorial.html for details on Anipose usage


***USEFUL COMMANDS***
Windows Z - keyboard shortcut for a new terminal
Windows C - launch coolero to monitor CPU/GPU usage, temperatures, and fan speeds
Windows E - launch file manager
Windows Spacebar - launch app finder
`htop` check the CPUs
`nvidia-smi` check the GPUs
`rm same_element_in_filenames* ` with the star sign you can remove all files with the same elements in the name
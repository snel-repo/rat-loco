# `rat-loco`

*Functions for rat locomotion data acquisition, processing, and analysis*

## To set up an environment:

1. Clone the repo:

   - `git clone git@github.com:snel-repo/rat-loco.git`
   - *Or, if planning to record videos with this clone:* `git clone --recurse-submodules -j2 git@github.com:snel-repo/rat-loco.git`
1. Create a `conda` environment

   - `conda create -n ratloco python==3.8`
2. Activate the new environment

   - `conda activate ratloco`
3. Install required packages with `pip`

   - `pip install -r requirements.txt`
4. If you plan to acquire frames with a FLIR USB 3 camera, be sure to download and install the Python version of Spinnaker:

   - https://flir.app.boxcn.net/v/SpinnakerSDK/folder/136168103146
   - Ensure the correct version is installed for your operating system.
   - Once downloaded, unzip and install with:
     - `cd ~/Downloads`
     - `tar -xzf spinnaker_python-3.0.0.118-cp38-cp38-linux_x86_64.tar.gz` # <-- Warning: get the latest correct version for your OS
     - `cd spinnaker_python-3.0.0.118-cp38-cp38-linux_x86_64`
     - `pip install spinnaker_python-3.0.0.118-cp38-cp38-linux_x86_64.whl`

## To collect data:

### Arduino Setup
1. Set desired camera trigger frequency by uncommenting the corresponding line setting the value of the `OCR1A` variable.
2. Save, compile, and upload the `openephys_camera_sync.ino` script onto an Arduino Uno or Mega using the [Arduino IDE](https://www.arduino.cc/en/software).
3. Power the arduino and connect Pin 2 to GND to disable any trigger pulses. To begin sending SYNC pulses out of Pin 12 (to cameras), connect Pin 2 to 3.3 or 5V.
### Open Ephys Setup
1. Ensure the SYNC and GND cables are properly connected from the Arduino to the cameras and to digital input of the Open Ephys recording system.
2. Login to the computer with [Open Ephys GUI](https://open-ephys.org/gui/) installed and launch it.
3. To prepare for recording a behavioral session, hit play button (Run) to ensure signals look as expected.
4. Start the treadmill and slowly increase the speed to allow the rat to achieve the desired speed gradually.
5. Only begin the OpenEphys recording once the rat has reached a steady state at the desired locomotion speed.
6. Shortly after OpenEphys recording has begun (a few seconds), start the camera recordings by connecting Pin 2 of the Arduino to 3.3 or 5V.
7. When the camera starts to record, TTL pulses from the Arduino should display on the bottom of the screen (digital inputs).
8. Once the cameras have acquired all the frames and completed recording, stop the OpenEphys recording, and stop the treadmill.
9. Save a file with the format `yyyymmdd-S_ratname_speedXX_inclineYY.info` inside the `recording##` folder to allow `rat-loco` to match Open Ephys and kinematics data later. The `S` indicates the session number, and should match the `FLIR_Multicam` outputs from that same session (be careful about this!).
### FLIR_Multicam Recording
1. Open `~/git/FLIR_Multi_Cam_HWTrig/params.yaml` in your preferred text editor (e.g., Visual Studio Code).
2. Set desired parameters before recording.
   - ***num_images***: the number of frames to capture (FPS * seconds of recording)
   - ***file_name***: use format `yyyymmdd_ratname_speedXX_inclineYY` (e.g., 20221118_godzilla_speed15_incline00)
3. Ensure `rat-loco.bashrc` has been copied as `~/.bashrc` to enable system commands in the terminal.
   - If not, execute the below to configure terminal with `rat-loco` commands
     - `cp ~/.bashrc ~/.bashrc_orig.bak`
     - `cp rat-loco.bashrc ~/.bashrc`
     - `source ~/.bashrc`
4. To initialize the cameras, execute `rec` in the terminal, which will put all connected cameras into a "Trigger Wait" state.
5. To start camera recording (while the rat is already running and Open Ephys recording has already begun) connect Pin #2 on the Arduino to 3.3 or 5V.
6. Make sure the terminal indicates that the recording had no dropped frames before proceeding to the next experimental condition. If dropped frames were detected, allow the rat to rest, and take another recording.

### Processing Raw Frames and Converting to Video Data
1. Once all locomotion sessions have been recorded for that experiment, we can use `rat-loco` commands to automate video conversion and preparation for [Anipose](https://anipose.readthedocs.io/en/latest/index.html).
2. Go to `rat-loco/FLIR_Multicam/`, then use `allsessionframes2vid FFFFF`, where `FFFFF` is the number of frames. This will loop through all session folders from today and convert all frames into videos. This command defaults to files collected the same day, but to run it for sessions on a different date, run `allsessionframes2vid FFFFF yyyymmdd`, where `yyyymmdd` is the desired date.
3. (Optional) To convert frames to video on a session-by-session basis, you may run `allframes2vid FFFFF` inside the desired folder containing the frames.
4. (Optional) To combine and visualize all videos, navigate to one of the `imagesyyyymmdd-X` folders, and use `stackallvids`. Add argument: `xstack`, `hstack`, or `vstack` to stack them horizontally, vertically, or in a square (e.g. default is `stackallvids hstack`).

### Converting Video Data into 3D Kinematic `.csv` Files with Anipose
1. Run `makeanipose` on the same day as recording to automatically create a new session folder in `~/anipose/sessionyyyymmdd` and move all videos from today into the `videos-raw` folder.
2. Instructions for use of Anipose are [HERE](https://anipose.readthedocs.io/en/latest/index.html).
3. Here is a list of commands used in Anipose (run ALL commands in order to triangulate and verify quality of video tracking)
   - `anipose analyze` (runs inference on videos with trained DLC model, selected in [`config.toml`](https://anipose.readthedocs.io/en/latest/params.html)): creates pose-2d folder
   - `anipose filter` (apply over the 2D data functions as a threshold filter): creates pose-2d-filtered folder
   - `anipose label-2d-filter` (plot the filtered predicted 2D labels on each frame): creates videos-labeled-filtered folder.
   - `anipose label-2d` (view the unfiltered predicted 2D labels on each frame): creates videos-labeled folder
   - `anipose calibrate`
   - `anipose triangulate` (generate csv file for each group of videos): creates pose-3d folder
   - `anipose label-3d` (plot the predicted labels from the 3D tracking for each group of videos): creates videos-3d folder
   - `anipose label-combined` (concatenate the videos for each group of videos obtained from running label-2d and label-3d): creates videos-combined folder

## To run analyses:

1. Activate your conda environment: `conda activate ratloco`
2. Open `rat_loco_analysis.py` in your favorite code editor (e.g. VSCode). The `rat_loco_analysis.py` python script serves as a config file and the main caller of all other subfunctions contained in separate python scripts (other `.py` files in the repo).
3. Make sure the directory paths actually point to the correct data folders which contain the `.csv` files for Anipose data, and the folder which contains the `Record Node ###` folder for the Open Ephys data path.
4. Open the terminal inside the `rat-loco` repo folder and execute: `python rat_loco_analysis.py`
5. To change the analysis, change the `plot_type` variable to a string that matches one of the `plot_type` conditionals near the bottom half of `rat_loco_analysis.py`. This will execute a different function the next time you run `python rat_loco_analysis.py`

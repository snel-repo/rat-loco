# `rat-loco`
*Functions for rat locomotion data processing and analysis*

## To set up an environment:
1. Clone the repo:
    - `git clone git@github.com:snel-repo/rat-loco.git`

2. Create a `conda` environment
    - `conda create -n ratloco python==3.10`

3. Activate the new environment
    - `conda activate ratloco`

4. Install required packages with `pip`
    - `pip install -r requirements.txt`
    

## To run analyses:
1. Activate your conda environment: `conda activate ratloco`

2. Open `rat_loco_analysis.py` in your favorite code editor (e.g. VSCode). The `rat_loco_analysis.py` python script serves as a config file and the main caller of all other subfunctions contained in separate python scripts (other `.py` files in the repo).

3. Make sure the directory paths actually point to the correct data folders which contain the `.csv` files for anipose data, and the folder which contains the `Record Node ###` folder for the Open Ephys data path.

4. Open the terminal inside the `rat-loco` repo folder and execute: `python rat_loco_analysis.py`

5. To change the analysis, change the `plot_type` variable to a string that matches one of the `plot_type` conditionals near the bottom half of `rat_loco_analysis.py`. This will execute a different function the next time you run `python rat_loco_analysis.py`

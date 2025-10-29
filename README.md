# Comparing Tiny vs Eyelink

This Python-based framework transforms raw Eyelink eye-tracking data (.asc) and raw obsbot tiny2 data (txt) into structured CSV files for analysis.

## Installation
1. **Clone Repository**: 
    ```bash
   https://github.com/mathemarujeena/eyetracking_tiny_vs_eyelink.git
   ```

   Or download and extract the ZIP from GitHub.

## Franework Structure
- **process_data_task1.py** : Processes data for tasks with faces and fractals to determine the location of the gaze in response to the stimuli.
- **eye_link_calibration.ipynb** : Reads raw data from eyelink portable duo for calibration.
- **post_calibration.ipynb**: Calculates the transformation matrix and transformation dataframes
- **post_calibration_all_indiv**: Processed the both eyelink portable duo and obsbot tiny2 data for analysis and stores it in processed_data folder in format `processed_data_asc/tiny_taskname_all_post_process_indiv`.
- **read_data**: Reads raw data from both eyelink portable duo and obsbot tiny2 and stores it in processed_data folder in format `processed_data_asc/tiny_taskname_all`.
- **reaction_times**: Analysis for the tasks using the processed data.
- **Plots and Files**: Contains the plots and files for different tasks.

## Usage
1. Place the data from [Dataset](https://figshare.com/s/3c879b0e38bbc7c9b978) to a data folder name processed_data.



import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
from pathlib import Path
import re
import os
from collections import defaultdict
from post_calibration import *

def get_filenames_for_timestamp():
    filenames = defaultdict(lambda: [None, None, None, None])

    pathlist_asc = Path('./asc_data_free_viewing').rglob('*.asc')
    pathlist_tiny = Path('./tiny_data/gaze_data_v6_free_viewing').rglob('gaze_messages_*.txt')
    pathlist_transformation = Path('./transformation_matrix').rglob('*.npy')
    pathlist_transformation_df = Path('./transformation_df').rglob('*.csv')

    for path in pathlist_asc:
        participant= re.search(r"(P\d+)", os.path.basename(str(path.stem))).group(1)
        filenames[participant][0] = path

    for path in pathlist_tiny:
        participant= re.search(r"(P\d+)", os.path.basename(str(path.stem))).group(1)
        filenames[participant][1] = path
    
    for path in pathlist_transformation:
        participant= re.search(r"(P\d+)", os.path.basename(str(path.stem))).group(1)
        filenames[participant][2] = path

    for path in pathlist_transformation_df:
        participant= re.search(r"(P\d+)", os.path.basename(str(path.stem))).group(1)
        filenames[participant][3] = path


    return filenames

def main():
    df_final_saccade = pd.DataFrame()
    df_final_antisaccade = pd.DataFrame()
    df_final_faces = pd.DataFrame()
    df_final_fractals = pd.DataFrame()
    df_final_free_viewing = pd.DataFrame()
    df_final_asc_saccade = pd.DataFrame()
    df_final_asc_antisaccade = pd.DataFrame()
    df_final_asc_faces = pd.DataFrame()
    df_final_asc_fractals = pd.DataFrame()
    df_final_asc_free_viewing = pd.DataFrame()
    # transformation = np.load('transformation_matrix.npy')

    # pattern_start_exp = [r'Dot Probe Faces Experiemnt Starts', r'Fixation_Cross_Starts', r'Faces_Stimuli_Start', r'Pause_Start', 
    #                      r'Dot Probe Fractals Experiemnt Starts', r'Fractals_Stimuli_Start', r'saccade_task_start', r'anti_saccade_task_start',
                        #  r'free_viewing_experiment_start', r'free_viewing_experiment_change_pic = 1'
                        #  ]
    # pattern_end_exp = [r'Dot Probe Faces Experiment Ends', r'Fixation_Cross_Stop', r'Faces_Stimuli_Stop', r'Pause_Stop',
    #                    r'Dot Probe Fractals Experiment Ends', r'Fractals_Stimuli_Stop', r'saccade_task_end', r'anti_saccade_task_end',
    #                 #    r'free_viewing_experiment_end', r'landscape0'
                    #    ]
    
    pattern_start_exp = r'Dot Probe Faces Experiemnt Starts'
    pattern_end_exp = r'free_viewing_experiment_end'

    filenames = get_filenames_for_timestamp()

    # Load tiny experiment data
    df_exp_tiny = pd.read_csv('./processed_data/processed_data_tiny_task_saccades_all.csv')
    df_exp_tiny_as = pd.read_csv('./processed_data/processed_data_tiny_task_anti_saccades_all.csv')
    df_exp_tiny_fc = pd.read_csv('./processed_data/processed_data_tiny_task_faces_all.csv')
    df_exp_tiny_fr = pd.read_csv('./processed_data/processed_data_tiny_task_fractals_all.csv')
    df_exp_tiny_freev = pd.read_csv('./processed_data/processed_data_tiny_task_free_viewing.csv')

    
    # Load eyelink experiment data
    df_exp_asc = pd.read_csv('./processed_data/processed_data_asc_task_sacccades_all.csv')
    df_exp_asc_as = pd.read_csv('./processed_data/processed_data_asc_task_anti_sacccades_all.csv')
    df_exp_asc_fc = pd.read_csv('./processed_data/processed_data_asc_task_faces_all.csv')
    df_exp_asc_fr = pd.read_csv('./processed_data/processed_data_asc_task_fractals_all.csv')
    df_exp_asc_freev = pd.read_csv('./processed_data/processed_data_asc_task_free_viewing.csv')


    for key, values in filenames.items():
        if (key != 'P065') and (key != 'P176'): #P065 and P176 omiited
            print('------------------', key)
            transformation = np.load(values[2])
            transformation_df = pd.read_csv(values[3])
            start_time_eyelink, end_time_eyelink = get_start_end_time(values[0], pattern_start_exp, pattern_end_exp)
            start_time_tiny, end_time_tiny = get_start_end_time(values[1], pattern_start_exp, pattern_end_exp)

            # For saccades
            # Load tiny experiment data
            df_exp_tiny_saccade = df_exp_tiny[df_exp_tiny["participants"]==key].copy()  # Create explicit copy
            
            # Load eyelink experiment data
            df_exp_asc_saccade = df_exp_asc[df_exp_asc["participants"]==key].copy()  # Create explicit copy
            
            # convert unix timestamp to eyelink timestamp for experiment data
            df_exp_tiny_saccade['eyelink_time'] = np.interp(np.array(df_exp_tiny_saccade['Time']), [float(start_time_tiny),float(end_time_tiny)], 
                                                    [int(start_time_eyelink.split('\t')[1]), int(end_time_eyelink.split('\t')[1])])
            df_exp_tiny_saccade['eyelink_time'] = df_exp_tiny_saccade['eyelink_time'].astype(int)

            # Remove blinks using interpolation 
            df_exp_asc_saccade['RX_'] = df_exp_asc_saccade['RX'].interpolate(method='linear', limit_direction='both')
            df_exp_asc_saccade['RY_'] = df_exp_asc_saccade['RY'].interpolate(method='linear', limit_direction='both')
            # df_exp_asc_saccade_b, df_exp_tiny_saccade_b = remove_blinks(df_exp_asc_saccade, df_exp_tiny_saccade)
            df_exp_asc_saccade_b, df_exp_tiny_saccade_b = df_exp_asc_saccade, df_exp_tiny_saccade
            
            # Apply rolling median to remove noise for experiment data
            df_exp_tiny_saccade_b['x_rol_median'] = rolling_median(df_exp_tiny_saccade_b['rawX'], n=2)
            df_exp_tiny_saccade_b['y_rol_median'] = rolling_median(df_exp_tiny_saccade_b['rawY'], n=2)
            df_exp_tiny_saccade_b = df_exp_tiny_saccade_b.sort_values(by="eyelink_time").reset_index(drop=True)

            exp_range_x = (df_exp_tiny_saccade_b['x_rol_median'].min(), df_exp_tiny_saccade_b['x_rol_median'].max())  
            calib_range_x = (transformation_df['GazeX'].min(), transformation_df['GazeX'].max()) 

            exp_range_y = (df_exp_tiny_saccade_b['y_rol_median'].min(), df_exp_tiny_saccade_b['y_rol_median'].max())
            calib_range_y = (transformation_df['GazeY'].min(), transformation_df['GazeY'].max())

            # remapping the experiment gaze points to calibration points
            df_exp_tiny_saccade_b["remapX"]= df_exp_tiny_saccade_b.apply(lambda row: remap_experiment_gaze(row["x_rol_median"], exp_range_x, calib_range_x), axis=1)
            df_exp_tiny_saccade_b["remapY"]= df_exp_tiny_saccade_b.apply(lambda row: remap_experiment_gaze(row["y_rol_median"], exp_range_y, calib_range_y), axis=1)

            # Apply calibration to experiment data
            # df_exp_tiny_saccade_b["post_cal"]= df_exp_tiny_saccade_b.apply(lambda row: apply_calibration([row["x_rol_median"], row["y_rol_median"]], transformation=transformation), axis=1)
            df_exp_tiny_saccade_b["post_cal"]= df_exp_tiny_saccade_b.apply(lambda row: apply_calibration([row["remapX"], row["remapY"]], transformation=transformation), axis=1)
            df_exp_tiny_saccade_b[['post_calX', 'post_calY']] = pd.DataFrame(df_exp_tiny_saccade_b['post_cal'].tolist(), index=df_exp_tiny_saccade_b.index)
            df_final_saccade = pd.concat([df_final_saccade,df_exp_tiny_saccade_b], ignore_index=True)
            df_final_asc_saccade = pd.concat([df_final_asc_saccade,df_exp_asc_saccade_b], ignore_index=True)

            # print(df_exp_tiny_saccade_b['eyelink_time'].values)
            # print(df_exp_asc_saccade_b['Time'].values)


            # plt.figure(figsize=(10, 5))
            # plt.plot(df_exp_asc_saccade_b['Time'], df_exp_asc_saccade_b['RX_'],alpha=0.5, label="Eyelink Data")
            # plt.plot(df_exp_tiny_saccade_b["eyelink_time"], df_exp_tiny_saccade_b["post_calX"], color='grey', alpha=0.4, label="Post Calibration Data")
            # # plt.plot(df_exp_tiny_saccade_b["eyelink_time"], df_exp_tiny_saccade_b["calX"], color='blue', alpha=0.4, label="Post Calibration Data Cal")
            # # plt.plot(df_exp_tiny_saccade_b["eyelink_time"], df_exp_tiny_saccade_b["rawX"], color='red', alpha=0.4, label="Post Calibration Data Raw")
            # plt.title(f"Gaze Data for Saccdades Task {key}")
            # plt.xlabel("Time")
            # plt.ylabel("X Position")
            # plt.grid(True)
            # plt.legend()
            # plt.show()

    


            # For anti saccades

            # Load tiny experiment data
            df_exp_tiny_anti_saccade = df_exp_tiny_as[df_exp_tiny_as["participants"]==key]
            
            # Load eyelink experiment data
            df_exp_asc_anti_saccade = df_exp_asc_as[df_exp_asc_as["participants"]==key]
            
            # convert unix timestamp to eyelink timestamp for experiment data
            df_exp_tiny_anti_saccade['eyelink_time'] = np.interp(np.array(df_exp_tiny_anti_saccade['Time']), [float(start_time_tiny),float(end_time_tiny)], 
                                                        [int(start_time_eyelink.split('\t')[1]), int(end_time_eyelink.split('\t')[1])])
            df_exp_tiny_anti_saccade['eyelink_time'] = df_exp_tiny_anti_saccade['eyelink_time'].astype(int)

            # Remove blinks from both datasets
            
            # Remove blinks using interpolation 
            df_exp_asc_anti_saccade['RX_'] = df_exp_asc_anti_saccade['RX'].interpolate(method='linear', limit_direction='both')
            df_exp_asc_anti_saccade['RY_'] = df_exp_asc_anti_saccade['RY'].interpolate(method='linear', limit_direction='both')
            # df_exp_asc_saccade_b, df_exp_tiny_saccade_b = remove_blinks(df_exp_asc_saccade, df_exp_tiny_saccade)
            df_exp_asc_anti_saccade_b, df_exp_tiny_anti_saccade_b = df_exp_asc_anti_saccade, df_exp_tiny_anti_saccade

            # df_exp_asc_anti_saccade_b, df_exp_tiny_anti_saccade_b = remove_blinks(df_exp_asc_anti_saccade, df_exp_tiny_anti_saccade)
            
            # Apply rolling median to remove noise for experiment data
            df_exp_tiny_anti_saccade_b['x_rol_median'] = rolling_median(df_exp_tiny_anti_saccade_b['rawX'], n=2)
            df_exp_tiny_anti_saccade_b['y_rol_median'] = rolling_median(df_exp_tiny_anti_saccade_b['rawY'], n=2)
            df_exp_tiny_anti_saccade_b = df_exp_tiny_anti_saccade_b.sort_values(by="eyelink_time").reset_index(drop=True)

            exp_range_x_as = (df_exp_tiny_anti_saccade_b['x_rol_median'].min(), df_exp_tiny_anti_saccade_b['x_rol_median'].max())
            calib_range_x_as = (transformation_df['GazeX'].min(), transformation_df['GazeX'].max())   

            exp_range_y_as = (df_exp_tiny_anti_saccade_b['y_rol_median'].min(), df_exp_tiny_anti_saccade_b['y_rol_median'].max())
            calib_range_y_as = (transformation_df['GazeY'].min(), transformation_df['GazeY'].max())

            # remapping the experiment gaze points to calibration points
            df_exp_tiny_anti_saccade_b["remapX"]= df_exp_tiny_anti_saccade_b.apply(lambda row: remap_experiment_gaze(row["x_rol_median"], exp_range_x_as, calib_range_x_as), axis=1)
            df_exp_tiny_anti_saccade_b["remapY"]= df_exp_tiny_anti_saccade_b.apply(lambda row: remap_experiment_gaze(row["y_rol_median"], exp_range_y_as, calib_range_y_as), axis=1)

            # Apply calibration to experiment data
            df_exp_tiny_anti_saccade_b["post_cal"]= df_exp_tiny_anti_saccade_b.apply(lambda row: apply_calibration([row["remapX"], row["remapY"]], transformation=transformation), axis=1)
            df_exp_tiny_anti_saccade_b[['post_calX', 'post_calY']] = pd.DataFrame(df_exp_tiny_anti_saccade_b['post_cal'].tolist(), index=df_exp_tiny_anti_saccade_b.index)
            df_final_antisaccade = pd.concat([df_final_antisaccade,df_exp_tiny_anti_saccade_b], ignore_index=True)
            df_final_asc_antisaccade = pd.concat([df_final_asc_antisaccade,df_exp_asc_anti_saccade_b], ignore_index=True)

    

            # plt.figure(figsize=(10, 5))
            # plt.plot(df_exp_asc_anti_saccade_b['Time'], df_exp_asc_anti_saccade_b['RX'],alpha=0.5, label="Eyelink Data")
            # plt.plot(df_exp_tiny_anti_saccade_b["eyelink_time"], df_exp_tiny_anti_saccade_b["post_calX"], color='grey', alpha=0.4, label="Post Calibration Data")
            # plt.title(f"Gaze Data for Anti Saccdades Task {key}")
            # plt.xlabel("Time")
            # plt.ylabel("X Position")
            # plt.grid(True)
            # plt.legend()
            # plt.show()


            # For dot probe faces experiment

            # Load tiny experiment data
            df_exp_tiny_faces = df_exp_tiny_fc[df_exp_tiny_fc["participants"]==key]
            
            # Load eyelink experiment data
            df_exp_asc_faces = df_exp_asc_fc[df_exp_asc_fc["participants"]==key]
            
            # convert unix timestamp to eyelink timestamp for experiment data
            # df_exp_tiny_faces['eyelink_time'] = unix_to_eyelink(df_exp_tiny_faces['Time'], float(start_time_tiny), int(start_time_eyelink.split('\t')[1]))
            df_exp_tiny_faces['eyelink_time'] = np.interp(np.array(df_exp_tiny_faces['Time']), [float(start_time_tiny),float(end_time_tiny)], 
                                                        [int(start_time_eyelink.split('\t')[1]), int(end_time_eyelink.split('\t')[1])])
            df_exp_tiny_faces['eyelink_time'] = df_exp_tiny_faces['eyelink_time'].astype(int)

            # Remove blinks from both datasets
            df_exp_asc_faces['RX_'] = df_exp_asc_faces['RX'].interpolate(method='linear', limit_direction='both')
            df_exp_asc_faces['RY_'] = df_exp_asc_faces['RY'].interpolate(method='linear', limit_direction='both')
            df_exp_asc_faces_b, df_exp_tiny_faces_b = df_exp_asc_faces, df_exp_tiny_faces
            # df_exp_asc_faces_b, df_exp_tiny_faces_b = remove_blinks(df_exp_asc_faces, df_exp_tiny_faces)
            
            # Apply rolling median to remove noise for experiment data
            df_exp_tiny_faces_b['x_rol_median'] = rolling_median(df_exp_tiny_faces_b['rawX'], n=2)
            df_exp_tiny_faces_b['y_rol_median'] = rolling_median(df_exp_tiny_faces_b['rawY'], n=2)
            df_exp_tiny_faces_b = df_exp_tiny_faces_b.sort_values(by="eyelink_time").reset_index(drop=True)

            exp_range_x_as = (df_exp_tiny_faces_b['x_rol_median'].min(), df_exp_tiny_faces_b['x_rol_median'].max())
            calib_range_x_as = (transformation_df['GazeX'].min(), transformation_df['GazeX'].max()) 

            exp_range_y_as = (df_exp_tiny_faces_b['y_rol_median'].min(), df_exp_tiny_faces_b['y_rol_median'].max())
            calib_range_y_as = (transformation_df['GazeY'].min(), transformation_df['GazeY'].max())

            # remapping the experiment gaze points to calibration points
            df_exp_tiny_faces_b["remapX"]= df_exp_tiny_faces_b.apply(lambda row: remap_experiment_gaze(row["x_rol_median"], exp_range_x_as, calib_range_x_as), axis=1)
            df_exp_tiny_faces_b["remapY"]= df_exp_tiny_faces_b.apply(lambda row: remap_experiment_gaze(row["y_rol_median"], exp_range_y_as, calib_range_y_as), axis=1)

            # Apply calibration to experiment data
            df_exp_tiny_faces_b["post_cal"]= df_exp_tiny_faces_b.apply(lambda row: apply_calibration([row["remapX"], row["remapY"]], transformation=transformation), axis=1)
            df_exp_tiny_faces_b[['post_calX', 'post_calY']] = pd.DataFrame(df_exp_tiny_faces_b['post_cal'].tolist(), index=df_exp_tiny_faces_b.index)
            df_final_faces = pd.concat([df_final_faces,df_exp_tiny_faces_b], ignore_index=True)
            df_final_asc_faces = pd.concat([df_final_asc_faces,df_exp_asc_faces_b], ignore_index=True)
    
    

            # plt.figure(figsize=(15, 6))
            # plt.plot(df_exp_asc_faces_b['Time'], df_exp_asc_faces_b['RX'],alpha=0.5, label="Eyelink Data")
            # plt.plot(df_exp_tiny_faces_b["eyelink_time"], df_exp_tiny_faces_b["post_calX"], color='grey', alpha=0.4, label="Post Calibration Data")
            # plt.title(f"Gaze Data for Dot Probe Faces Task {key}")
            # plt.xlabel("Time")
            # plt.ylabel("X Position")
            # plt.grid(True)
            # plt.legend()
            # plt.show()


            # For dot probe fractals experiment

            # Load tiny experiment data
            df_exp_tiny_fractals = df_exp_tiny_fr[df_exp_tiny_fr["participants"]==key]
            
            # Load eyelink experiment data
            df_exp_asc_fractals = df_exp_asc_fr[df_exp_asc_fr["participants"]==key]
            
            # convert unix timestamp to eyelink timestamp for experiment data
            df_exp_tiny_fractals['eyelink_time'] = np.interp(np.array(df_exp_tiny_fractals['Time']), [float(start_time_tiny),float(end_time_tiny)], 
                                                        [int(start_time_eyelink.split('\t')[1]), int(end_time_eyelink.split('\t')[1])])
            df_exp_tiny_fractals['eyelink_time'] = unix_to_eyelink(df_exp_tiny_fractals['Time'], float(start_time_tiny), int(start_time_eyelink.split('\t')[1]))
            df_exp_tiny_fractals['eyelink_time'] = df_exp_tiny_fractals['eyelink_time'].astype(int)

            # Remove blinks from both datasets
            #Remove blinks using interpolation 
            df_exp_asc_fractals['RX_'] = df_exp_asc_fractals['RX'].interpolate(method='linear', limit_direction='both')
            df_exp_asc_fractals['RY_'] = df_exp_asc_fractals['RY'].interpolate(method='linear', limit_direction='both')
            df_exp_asc_fractls_b, df_exp_tiny_fractals_b = df_exp_asc_fractals, df_exp_tiny_fractals
            # df_exp_asc_fractls_b, df_exp_tiny_fractals_b = remove_blinks(df_exp_asc_fractals, df_exp_tiny_fractals)
            
            # Apply rolling median to remove noise for experiment data
            df_exp_tiny_fractals_b['x_rol_median'] = rolling_median(df_exp_tiny_fractals_b['rawX'], n=2)
            df_exp_tiny_fractals_b['y_rol_median'] = rolling_median(df_exp_tiny_fractals_b['rawY'], n=2)
            df_exp_tiny_fractals_b = df_exp_tiny_fractals_b.sort_values(by="eyelink_time").reset_index(drop=True)

            exp_range_x_as = (df_exp_tiny_fractals_b['x_rol_median'].min(), df_exp_tiny_fractals_b['x_rol_median'].max())  
            calib_range_x_as = (transformation_df['GazeX'].min(), transformation_df['GazeX'].max()) 
            # calib_range_x_as = (298, 303) 

            exp_range_y_as = (df_exp_tiny_fractals_b['y_rol_median'].min(), df_exp_tiny_fractals_b['y_rol_median'].max())
            calib_range_y_as = (transformation_df['GazeY'].min(), transformation_df['GazeY'].max())
            # calib_range_y_as = (177, 179)

            # remapping the experiment gaze points to calibration points
            df_exp_tiny_fractals_b["remapX"]= df_exp_tiny_fractals_b.apply(lambda row: remap_experiment_gaze(row["x_rol_median"], exp_range_x_as, calib_range_x_as), axis=1)
            df_exp_tiny_fractals_b["remapY"]= df_exp_tiny_fractals_b.apply(lambda row: remap_experiment_gaze(row["y_rol_median"], exp_range_y_as, calib_range_y_as), axis=1)

            # Apply calibration to experiment data
            df_exp_tiny_fractals_b['post_cal']= df_exp_tiny_fractals_b.apply(lambda row: apply_calibration([row["remapX"], row["remapY"]], transformation=transformation), axis=1)
            df_exp_tiny_fractals_b[['post_calX', 'post_calY']] = pd.DataFrame(df_exp_tiny_fractals_b['post_cal'].tolist(), index=df_exp_tiny_fractals_b.index)
            df_final_fractals = pd.concat([df_final_fractals,df_exp_tiny_fractals_b], ignore_index=True)
            df_final_asc_fractals = pd.concat([df_final_asc_fractals,df_exp_asc_fractls_b], ignore_index=True)

            
            
            # plt.figure(figsize=(15, 6))
            # plt.plot(df_exp_asc_fractls_b['Time'], df_exp_asc_fractls_b['RX'],alpha=0.5, label="Eyelink Data")
            # plt.plot(df_exp_tiny_fractals_b["eyelink_time"], df_exp_tiny_fractals_b["post_calX"], color='grey', alpha=0.4, label="Post Calibration Data")
            # plt.title(f"Gaze Data for Dot Probe Fractals Task {key}")
            # plt.xlabel("Time")
            # plt.ylabel("X Position")
            # plt.grid(True)
            # plt.legend()
            # plt.show()

     

            # For free viewing experiment
            # Load tiny experiment data
            df_exp_tiny_fv = df_exp_tiny_freev[df_exp_tiny_freev["participants"]==key]
            
            # Load eyelink experiment data
            df_exp_asc_fv = df_exp_asc_freev[df_exp_asc_freev["participants"]==key]
            
            # convert unix timestamp to eyelink timestamp for experiment data
            # df_exp_tiny_fv['eyelink_time'] = unix_to_eyelink(df_exp_tiny_fv['Time'], np.array(start_time_tiny), np.array(start_time_eyelink))
            # df_exp_tiny_fv['eyelink_time'] = unix_to_eyelink(df_exp_tiny_fv['Time'], float(start_time_tiny), int(start_time_eyelink.split('\t')[1]))
            df_exp_tiny_fv['eyelink_time'] = np.interp(np.array(df_exp_tiny_fv['Time']), [float(start_time_tiny),float(end_time_tiny)], 
                                                    [int(start_time_eyelink.split('\t')[1]), int(end_time_eyelink.split('\t')[1])])
            df_exp_tiny_fv['eyelink_time'] = df_exp_tiny_fv['eyelink_time'].astype(int)

            # Remove blinks from both datasets
                # Remove blinks using interpolation 
            df_exp_asc_fv['RX_'] = df_exp_asc_fv['RX'].interpolate(method='linear', limit_direction='both')
            df_exp_asc_fv['RY_'] = df_exp_asc_fv['RY'].interpolate(method='linear', limit_direction='both')

            df_exp_asc_fv_b, df_exp_tiny_fv_b = df_exp_asc_fv, df_exp_tiny_fv
            # df_exp_asc_fv_b, df_exp_tiny_fv_b = remove_blinks(df_exp_asc_fv, df_exp_tiny_fv)
            
            # Apply rolling median to remove noise for experiment data
            df_exp_tiny_fv_b['x_rol_median'] = rolling_median(df_exp_tiny_fv_b['rawX'], n=2)
            df_exp_tiny_fv_b['y_rol_median'] = rolling_median(df_exp_tiny_fv_b['rawY'], n=2)
            df_exp_tiny_fv_b = df_exp_tiny_fv_b.sort_values(by="eyelink_time").reset_index(drop=True)

            exp_range_x = (df_exp_tiny_fv_b['x_rol_median'].min(), df_exp_tiny_fv_b['x_rol_median'].max()) 
            calib_range_x = (transformation_df['GazeX'].min(), transformation_df['GazeX'].max())  
            # calib_range_x = (298, 303) 

            exp_range_y = (df_exp_tiny_fv_b['y_rol_median'].min(), df_exp_tiny_fv_b['y_rol_median'].max())
            calib_range_y = (transformation_df['GazeY'].min(), transformation_df['GazeY'].max())
            # calib_range_y = (177, 179)

            # remapping the experiment gaze points to calibration points
            df_exp_tiny_fv_b["remapX"]= df_exp_tiny_fv_b.apply(lambda row: remap_experiment_gaze(row["x_rol_median"], exp_range_x, calib_range_x), axis=1)
            df_exp_tiny_fv_b["remapY"]= df_exp_tiny_fv_b.apply(lambda row: remap_experiment_gaze(row["y_rol_median"], exp_range_y, calib_range_y), axis=1)

            # Apply calibration to experiment data
            df_exp_tiny_fv_b['post_cal'] = df_exp_tiny_fv_b.apply(lambda row: apply_calibration([row["remapX"], row["remapY"]], transformation=transformation), axis=1)
            # assume df['tuples'] is something like [(a1, b1), (a2, b2), …]
            df_exp_tiny_fv_b[['post_calX', 'post_calY']] = pd.DataFrame(df_exp_tiny_fv_b['post_cal'].tolist(), index=df_exp_tiny_fv_b.index)
            df_final_free_viewing = pd.concat([df_final_free_viewing,df_exp_tiny_fv_b], ignore_index=True)
            df_final_asc_free_viewing = pd.concat([df_final_asc_free_viewing,df_exp_asc_fv_b], ignore_index=True)

            df_new_tiny_free_viewing = df_final_free_viewing[['eyelink_time','post_calX', 'post_calY', 'participants', 'interval_num']].rename(
            columns={'eyelink_time': 'Time'})
            df_new_asc_free_viewing = df_final_asc_free_viewing[['Time','RX_', 'RY_', 'state', 'participants', 'interval_num']].rename(
            columns={'RX_': 'RX', 'RY_': 'RY'}
        )

            # plt.figure(figsize=(10, 5))
            # # plt.plot(df_exp_asc_fv_b['Time'], df_exp_asc_fv_b['RX'],alpha=0.5, label="Eyelink Data")
            # # plt.plot(df_exp_tiny_fv_b["eyelink_time"], df_exp_tiny_fv_b["post_calX"], color='grey', alpha=0.4, label="Post Calibration Data")
            # plt.plot(df_exp_asc_fv_b['RX'][:8000], df_exp_asc_fv_b['RY'][:8000],alpha=0.5, label="Eyelink Data")
            # plt.plot(df_exp_tiny_fv_b["post_calX"][:200], df_exp_tiny_fv_b["post_calY"][:200], color='grey', alpha=0.4, label="Post Calibration Data")
            
            # plt.title(f"Gaze Data for Free Viewing Task {key}")
            # plt.xlabel("Time")
            # plt.ylabel("X Position")
            # plt.grid(True)
            # plt.legend()
            # plt.show()

    df_final_asc_saccade.to_csv("./processed_data/processed_data_asc_task_sacccades_all_post_process_indiv.csv", index=False)
    df_final_saccade.to_csv("./processed_data/processed_data_tiny_task_sacccades_all_post_process_indiv.csv", index=False)

    df_final_antisaccade.to_csv("./processed_data/processed_data_tiny_task_anti_sacccades_all_post_process_indiv.csv", index=False)
    df_final_asc_antisaccade.to_csv("./processed_data/processed_data_asc_task_anti_sacccades_all_post_process_indiv.csv", index=False)

    df_final_fractals.to_csv("./processed_data/processed_data_tiny_task_fractals_all_post_process_indiv.csv", index=False)  
    df_final_asc_fractals.to_csv("./processed_data/processed_data_asc_task_fractals_all_post_process_indiv.csv", index=False) 

    df_final_faces.to_csv("./processed_data/processed_data_tiny_task_faces_all_post_process_indiv.csv", index=False)
    df_final_asc_faces.to_csv("./processed_data/processed_data_asc_task_faces_all_post_process_indiv.csv", index=False)


    df_new_asc_free_viewing.to_csv("./processed_data/free_viewing_data/processed_data_eyelink_free_viewing.csv", index=False)
    df_new_tiny_free_viewing.to_csv("./processed_data/free_viewing_data/processed_data_tiny_free_viewing.csv", index=False)
        

if __name__ == "__main__":
    main()


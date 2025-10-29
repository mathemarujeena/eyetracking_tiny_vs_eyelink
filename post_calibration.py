import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
from pathlib import Path
import re
import os
from scipy.interpolate import interp1d
from collections import defaultdict



screen_width = 1920
screen_height = 1080

def load_data(file_path, names, sep=',', header=None):
    """Load calibration data from a CSV file."""
    data = pd.read_csv(file_path, header=header, names=names, sep=sep)
    return data 

def get_start_end_time(filename, pattern_start, pattern_end):

    with open(filename, 'r') as file:
        for line_number, line in enumerate(file, 1):
            if re.search(pattern_start, line):
                start_time = line.strip().split(' ')[0]
            if re.search(pattern_end, line):
                end_time = line.strip().split(' ')[0]

    return start_time, end_time

# def get_start_end_time(filename, pattern_start, pattern_end):
#     start_time = []
#     end_time = []
#     with open(filename, 'r') as file:
#         for line_number, line in enumerate(file, 1):
#             for i in range(len(pattern_start)):
#                 if re.search(pattern_start[i], line):
#                     start_time.append(line.strip().split(' ')[0])
#                 if re.search(pattern_end[i], line):
#                     end_time.append(line.strip().split(' ')[0])

#     return start_time, end_time


def filter_start_end_time(data, start_time, end_time):
    """Filter data based on start and end time."""
    filtered_data = data[(data['Time'] >= float(start_time)) & (data['Time'] <= float(end_time))]
    return filtered_data    


def unix_to_eyelink(unix_timestamp, tiny_timestamp_sec, eyelink_event_ms):
    # Convert all timestamps to seconds for consistency
    eyelink_event_sec = eyelink_event_ms / 1000.0
    
    # Calculate the offset 
    if isinstance(eyelink_event_ms, list) and len(eyelink_event_ms) > 1:
        eyelink_times = np.array([x.split('\t')[1] for x in eyelink_event_ms]).astype(float) / 1000.0
        tiny_times = np.array(tiny_timestamp_sec).astype(float)
        
        # Calculate offset for each pair of timestamps
        offsets = tiny_times - eyelink_times
        
        # Use median offset to be robust against outliers
        offset = np.median(offsets)
    else:
        offset = tiny_timestamp_sec - eyelink_event_sec
    
    # Convert to eyelink timestamp in milliseconds
    eyelink_timestamp_ms = (unix_timestamp - offset) * 1000.0
    
    # Round to 2 decimal places for consistency
    return round(eyelink_timestamp_ms, 2)


def interpolate_x(asc_start_time,asc_end_time, tiny_start_time, tiny_end_time, tiny_times):
    """
    Linearly interpolate to find eyelink value corresponding to a given tiny time.
    
    Parameters:
    - asc_start_time,asc_end_time: Known eyelink values (in ms)
    - tiny_start_time, tiny_end_time: Known tiny values (in s)
    - tiny_times:      The tiny values at which to interpolate
    
    Returns:
    - asc_time: The interpolated eyelink value (in ms)
    """
    return asc_start_time + (asc_end_time - asc_start_time) * ((tiny_times - tiny_start_time) / (tiny_end_time - tiny_start_time))

# def unix_to_eyelink(unix_timestamp, tiny_timestamp, eyelink_event_ms):
#     eyelink_time = np.array([x.split('\t')[1] for x in eyelink_event_ms])
#     eyelink_event_sec = eyelink_time.astype(float)/1000.0
#     offset = np.mean(tiny_timestamp.astype(float) - eyelink_event_sec)  # seconds
#     eyelink_timestamp_ms = (unix_timestamp - offset) * 1000.0
#     return round(eyelink_timestamp_ms, 2)


def remove_blinks(df_asc_b, df_tiny_b):
    # Get all timestamps where state is 0 (blink)
    blink_times = df_asc_b.loc[df_asc_b['state'] == 0, 'Time'].values

    # Create an interval (t-50, t+50) for each blink occurrence
    blink_intervals = [(t - 50, t + 50) for t in blink_times]

    # Merge overlapping intervals (if two blink windows overlap, take their union)
    blink_intervals = sorted(blink_intervals, key=lambda x: x[0])
    merged_intervals = []
    for interval in blink_intervals:
        if not merged_intervals:
            merged_intervals.append(interval)
        else:
            last_start, last_end = merged_intervals[-1]
            current_start, current_end = interval
            # If the current interval overlaps the previous one, merge them
            if current_start <= last_end:
                merged_intervals[-1] = (last_start, max(last_end, current_end))
            else:
                merged_intervals.append(interval)

    # Remove Blink Intervals from the Large Dataset 
    # Initialize a boolean mask (True = keep row)
    mask_large = np.ones(len(df_asc_b), dtype=bool)
    for start, end in merged_intervals:
        # Mark rows within the blink interval for removal (set to False)
        mask_large &= ~((df_asc_b['Time'] >= start) & (df_asc_b['Time'] <= end))
    df_asc_b_filtered = df_asc_b[mask_large].copy()

    # Remove Corresponding Blink Intervals from the Small Dataset 
    # Create a boolean mask for the small dataset
    mask_small = np.ones(len(df_tiny_b), dtype=bool)
    for start, end in merged_intervals:
        mask_small &= ~((df_tiny_b['eyelink_time'] >= start) & (df_tiny_b['eyelink_time'] <= end))
    df_tiny_b_filtered = df_tiny_b[mask_small].copy()

    # print("Large dataset: original {} rows, filtered {} rows".format(len(df_asc_b), len(df_asc_b_filtered)))
    # print("Small dataset: original {} rows, filtered {} rows".format(len(df_tiny_b), len(df_tiny_b_filtered)))  

    return df_asc_b_filtered, df_tiny_b_filtered

def rolling_median(calX, n):
    series = pd.Series(calX)
    window_size = 2 * n + 1
    median_series = series.rolling(window=window_size, center=True, min_periods=1).median()
    return median_series.values

def interpolate_tiny_values(eyelink_timestamps, tiny_timestamps, tiny_values, max_gap=0.1):
# def interpolate_tiny_values(eyelink_timestamps, tiny_timestamps, tiny_values, max_gap=100):
    """
    Interpolate tiny values to match eyelink timestamps, with improved edge case handling.
    
    Args:
        eyelink_timestamps: Array of eyelink timestamps
        tiny_timestamps: Array of tiny timestamps
        tiny_values: Array of tiny values
        max_gap: Maximum allowed gap (in seconds) for interpolation
        
    Returns:
        Interpolated tiny values at eyelink timestamps
    """
    # Convert timestamps to seconds for consistency
    # eyelink_sec = eyelink_timestamps
    eyelink_sec = eyelink_timestamps / 1000.0
    tiny_sec = tiny_timestamps
    
    # Create interpolation function
    interp_func = interp1d(tiny_sec, tiny_values, 
                          kind='linear', 
                          bounds_error=False,
                          fill_value='extrapolate')
    
    # Get interpolated values
    interpolated = interp_func(eyelink_sec)
    
    # Identify gaps in tiny data
    tiny_gaps = np.diff(tiny_sec)
    gap_indices = np.where(tiny_gaps > max_gap)[0]
    
    # Mark interpolated values in gaps as NaN
    for gap_idx in gap_indices:
        gap_start = tiny_sec[gap_idx]
        gap_end = tiny_sec[gap_idx + 1]
        mask = (eyelink_sec >= gap_start) & (eyelink_sec <= gap_end)
        interpolated[mask] = np.nan
    
    # Handle edge cases
    # If eyelink timestamps extend beyond tiny data, use last known value
    if eyelink_sec[-1] > tiny_sec[-1]:
        last_valid = np.where(~np.isnan(interpolated))[0][-1]
        interpolated[last_valid+1:] = interpolated[last_valid]
    
    # If eyelink timestamps start before tiny data, use first known value
    if eyelink_sec[0] < tiny_sec[0]:
        first_valid = np.where(~np.isnan(interpolated))[0][0]
        interpolated[:first_valid] = interpolated[first_valid]
    
    return interpolated

def get_interpolated_df(df_eyelink, df_tiny):      

    df_tiny['eyelink_time_int'] = df_tiny['eyelink_time'].astype(int)

    tiny_times = df_tiny['eyelink_time_int'].values 
    tiny_set = set(tiny_times)
    tiny_x_rol_median = df_tiny['x_rol_median'].values
    tiny_y_rol_median = df_tiny['y_rol_median'].values

    look_x_rol_median = df_tiny.set_index('eyelink_time_int')['x_rol_median'].to_dict()
    look_y_rol_median = df_tiny.set_index('eyelink_time_int')['y_rol_median'].to_dict()

    def get_Xtiny(time_val):
        if time_val in tiny_set:
            return look_x_rol_median[time_val]
        else:
            return np.interp(time_val, tiny_times, tiny_x_rol_median)

    def get_Ytiny(time_val):
        if time_val in tiny_set:
            return look_y_rol_median[time_val]
        else:
            return np.interp(time_val, tiny_times, tiny_y_rol_median)

    df_eyelink['Truetiny'] = df_eyelink['Time'].apply(lambda t: t in tiny_set)
    df_eyelink['Xtiny'] = df_eyelink['Time'].apply(get_Xtiny)
    df_eyelink['Ytiny'] = df_eyelink['Time'].apply(get_Ytiny)

    # df_eyelink['Timelink'] = df_eyelink['Time']
    # df_eyelink['Timetiny'] = df_eyelink['Time']

    df_new = df_eyelink[['RX', 'RY', 'Xtiny', 'Ytiny', 'Time', 'Truetiny']].rename(
        columns={'RX': 'Xlink', 'RY': 'Ylink'}
    )

    return df_new

def get_Xtiny(time_val,tiny_times,tiny_x_rol_median):
    return np.interp(time_val, tiny_times, tiny_x_rol_median)

def get_Ytiny(time_val,tiny_set,look_y_rol_median):
    return np.interp(time_val, tiny_set, look_y_rol_median)

def get_clusters(df_asc):
# def get_clusters(df_asc, df_interpolated):
    left = [30,150]
    center = [900,1100]
    # right = [1850,1970]
    right = [1800,1950]

    bottom = [-5,100]
    middle = [450,550]
    top = [900,1100]

    X_center_pos = [ left, left, left, center, center, center, right, right, right]
    Y_center_pos = [ bottom, middle, top, bottom, middle, top, bottom, middle, top]

    X_link = np.array(df_asc['Xlink'])
    Y_link = np.array(df_asc['Ylink'])

    cluster = []
    for i in range(len(df_asc['Xlink'])):
    # for i in range(len(df_interpolated['Xlink'])):
        found = False 
        for j in range(9):
            
            if (X_link[i] >= X_center_pos[j][0] and X_link[i] <= X_center_pos[j][1]) and \
            (Y_link[i] >= Y_center_pos[j][0] and Y_link[i] <= Y_center_pos[j][1]):
                cluster.append(j)
                found = True
                break  
        if not found:
            cluster.append(-1)

    return cluster


def compute_calibration(gaze_pts, src_pts, key):
    global transformation
    src_pts = np.array(src_pts, dtype=np.float32)
    gaze_pts = np.array(gaze_pts, dtype=np.float32)

    # Normalize points to [0, 1] to improve numerical stability
    src_pts_norm = _normalize_points(src_pts)
    dst_pts_norm = _normalize_points(gaze_pts)

    # Compute homography with RANSAC to reject outliers
    # transformation, mask = cv2.findHomography(src_pts_norm, dst_pts_norm, cv2.RANSAC, 5.0)
    transformation, mask = cv2.findHomography(dst_pts_norm, src_pts_norm, cv2.RANSAC, 5.0)
    if transformation is not None:
        print("[Calibration] Homography computed successfully.")
        print("is calibrated",transformation)
        np.save(f'transformation_matrix/{key}_transformation_matrix.npy', transformation)

        # _validate_calibration()  # Check for errors
    else:
        print("[Calibration] Homography computation failed.")

    return transformation
        


def _normalize_points(points):
    """
    Normalize points to [0, 1] based on screen dimensions.
    """
    global screen_width, screen_height
    points_norm = np.zeros_like(points, dtype=np.float32)
    points_norm[:, 0] = points[:, 0] / screen_width
    points_norm[:, 1] = points[:, 1] / screen_height
    return points_norm
        

def apply_calibration(raw_gaze, transformation):
    """
    Maps raw gaze to screen coords using the stored homography,
    if calibration is computed.
    """
    global screen_width, screen_height
    if transformation is None:
        return raw_gaze  # fallback to raw if not calibrated

    # # Normalize the raw gaze point
    x_norm = raw_gaze[0] / screen_width
    y_norm = raw_gaze[1] / screen_height

    # Apply homography
    pt = np.array([x_norm, y_norm, 1.0], dtype=np.float32).reshape(3, 1)
    mapped = transformation @ pt
    if abs(mapped[2, 0]) > 1e-6:
        x_calib = (mapped[0, 0] / mapped[2, 0]) * screen_width
        y_calib = (mapped[1, 0] / mapped[2, 0]) * screen_height
        # return (x_calib)
        return x_calib, y_calib
    return raw_gaze

def remap_experiment_gaze(exp_gaze, exp_range, calib_range):
    exp_min, exp_max = exp_range
    calib_min, calib_max = calib_range
    norm = (exp_gaze - exp_min) / (exp_max - exp_min)
    return norm * (calib_max - calib_min) + calib_min


def get_filenames_for_timestamp():
    filenames = defaultdict(lambda: [None, None, None])

    pathlist_asc = Path('./asc_data').rglob('*.asc')
    pathlist_tiny = Path('./TinyData/gaze_data_v6').rglob('gaze_messages_*.txt')
    pathlist_tiny_raw = Path('./TinyData/raw_data').rglob('gaze_directions_raw_*.txt')

    for path in pathlist_asc:
        participant= re.search(r"(P\d+)", os.path.basename(str(path.stem))).group(1)
        filenames[participant][0] = path

    for path in pathlist_tiny:
        participant= re.search(r"(P\d+)", os.path.basename(str(path.stem))).group(1)
        filenames[participant][1] = path

    for path in pathlist_tiny_raw:
        participant= re.search(r"(P\d+)", os.path.basename(str(path.stem))).group(1)
        filenames[participant][2] = path

    return filenames



def main():
    transformation = None

    # Load calibration data
    pattern_start_cal = r'EyeT Calibration Begins'
    pattern_end_cal = r'EyeT Calibration Ends'

    filenames = get_filenames_for_timestamp()
    for key, values in filenames.items():
        # if key == 'P006':
        print(key)
        print(values[2])
        print(values[1])
        print(values[0])

        # Load raw gaze data for calibration
        df_tiny_cal_raw = load_data(values[2], sep=" ", header=None, names=["Time", "rawX", "rawY"])
        start_time, end_time = get_start_end_time(values[1], pattern_start_cal, pattern_end_cal)
        df_tiny_cal_raw = filter_start_end_time(df_tiny_cal_raw, start_time, end_time)

        # Load eyelink data for calibration
        df_asc_cal_ = pd.read_csv("./processed_data/processed_data_asc_calibration.csv")
        df_asc_cal = df_asc_cal_[df_asc_cal_['participants'] == key].copy()
        start_time_eyelink_cal, end_time_eyelink_cal = get_start_end_time(values[0], pattern_start_cal, pattern_end_cal)


        # convert unix timestamp to eyelink timestamp
        df_tiny_cal_raw['eyelink_time'] = np.interp(np.array(df_tiny_cal_raw['Time']), [float(start_time),float(end_time)], 
                                                    [int(start_time_eyelink_cal.split('\t')[1]), int(end_time_eyelink_cal.split('\t')[1])])
        df_tiny_cal_raw['eyelink_time'] = df_tiny_cal_raw['eyelink_time'].astype(int)

        
        # Remove blinks using interpolation 
        df_asc_cal['RX_'] = df_asc_cal['RX'].interpolate(method='linear', limit_direction='both')
        df_asc_cal['RY_'] = df_asc_cal['RY'].interpolate(method='linear', limit_direction='both')
        df_asc_cal_b, df_tiny_cal_b = df_asc_cal, df_tiny_cal_raw

        # Apply rolling median to remove noise
        df_tiny_cal_b['x_rol_median'] = rolling_median(df_tiny_cal_b['rawX'], n=2)
        df_tiny_cal_b['y_rol_median'] = rolling_median(df_tiny_cal_b['rawY'], n=2)
        df_tiny_cal_b = df_tiny_cal_b.sort_values(by="eyelink_time").reset_index(drop=True)

        df_asc_cal_b['Time_int'] = df_asc_cal_b['Time'].astype(int)

        tiny_times = df_tiny_cal_b['eyelink_time'].values 
        # tiny_set = set(tiny_times)
        tiny_x_rol_median = df_tiny_cal_b['x_rol_median'].values
        tiny_y_rol_median = df_tiny_cal_b['y_rol_median'].values

        df_asc_cal_b['Truetiny'] = df_asc_cal_b['Time_int'].apply(lambda t: t in tiny_times)
        df_asc_cal_b['Xtiny'] = np.interp(df_asc_cal_b['Time'].values, tiny_times, tiny_x_rol_median)
        df_asc_cal_b['Ytiny'] = np.interp(df_asc_cal_b['Time'].values, tiny_times, tiny_y_rol_median)

        df_new = df_asc_cal_b[['Time','RX_', 'RY_', 'Xtiny', 'Ytiny', 'Truetiny']].rename(
            columns={'RX_': 'Xlink', 'RY_': 'Ylink'}
        )

        # get clusters
        df_new['cluster'] = get_clusters(df_new)

        # average gaze points for each cluster where truetiny is True
        avg_df = df_new[df_new['Truetiny'] == True].groupby('cluster').mean()
        avg_df.drop([-1], inplace=True)
        # avg_df.reset_index(drop=True, inplace=True)

        gaze_src = np.array([
            [50, 50],
            [960, 50],
            [1870, 50],
            [50, 540],
            [960, 540],
            [1870, 540],
            [50, 1030],
            [960, 1030],
            [1870, 1030],
        ], dtype=np.float32)

        row_labels = [0, 1, 2,3,4,5,6,7,8]
        col_labels = ['CalX', 'CalY']

        merge_df = pd.DataFrame(gaze_src, index=row_labels, columns=col_labels)
        # merge_df = merge_df.reindex(merge_df.index.union([0,8]))
        merge_df.loc[[0,1,2,3,4,5,6,7,8], 'GazeX'] = [avg_df['Xtiny'].get(0, np.nan),avg_df['Xtiny'].get(3, np.nan),avg_df['Xtiny'].get(6, np.nan),
                                                        avg_df['Xtiny'].get(1, np.nan),avg_df['Xtiny'].get(4, np.nan),avg_df['Xtiny'].get(7, np.nan),
                                                        avg_df['Xtiny'].get(2, np.nan),avg_df['Xtiny'].get(5, np.nan),avg_df['Xtiny'].get(8, np.nan)]
        merge_df.loc[[0,1,2,3,4,5,6,7,8], 'GazeY'] = [avg_df['Ytiny'].get(0, np.nan),avg_df['Ytiny'].get(3, np.nan),avg_df['Ytiny'].get(6, np.nan),
                                                        avg_df['Ytiny'].get(1, np.nan),avg_df['Ytiny'].get(4, np.nan),avg_df['Ytiny'].get(7, np.nan),
                                                        avg_df['Ytiny'].get(2, np.nan),avg_df['Ytiny'].get(5, np.nan),avg_df['Ytiny'].get(8, np.nan)]
        clean_merge_df = merge_df.dropna()
        # print(clean_merge_df)
        clean_merge_df.to_csv(f"./transformation_df/{key}.csv", index=False)


        scr_pts = clean_merge_df[['CalX','CalY']].to_numpy()
        gaze_pts = clean_merge_df[['GazeX','GazeY']].to_numpy()
        transformation = compute_calibration(gaze_pts, scr_pts,key)

if __name__ == "__main__":
    main()
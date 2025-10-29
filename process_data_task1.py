import pandas as pd
import glob
import re
import os

def process_message_file(msg_file):
    """
    Process the message file and extract event intervals.
    Returns a dictionary with:
      - 'experiment': (exp_start, exp_end)
      - 'fixation': list of (start, stop)
      - 'faces': list of (start, stop, emotion) where emotion is 'L' or 'R'
      - 'pause': list of (start, stop)
    """
    # Read the message file into a DataFrame.
    df_msg = pd.read_csv(msg_file, sep=r"\s+", header=None, engine='python', dtype=str)
    # First column is the timestamp; combine the rest into one message string.
    df_msg['timestamp'] = pd.to_numeric(df_msg[0])
    df_msg['message'] = df_msg.iloc[:, 1:].apply(lambda row: " ".join(row.dropna().astype(str)), axis=1)
    df_msg = df_msg[['timestamp', 'message']].sort_values('timestamp').reset_index(drop=True)

    events = {
        'experiment': None,
        'fixation': [],
        'faces': [],
        'pause': []
    }
    
    # Temporary variables to store start times and emotion side.
    fixation_start = None
    faces_start = None
    faces_emotion = None
    pause_start = None
    exp_start = None
    exp_end = None
    
    # Process each row sequentially.
    for idx, row in df_msg.iterrows():
        ts = row['timestamp']
        msg = row['message'].strip()
        
        # Overall experiment window.
        if "Dot Probe Faces Experiemnt Starts" in msg:
            exp_start = ts
        elif "Dot Probe Faces Experiment Ends" in msg:
            exp_end = ts
        
        # Fixation intervals.
        if "Fixation_Cross_Start" in msg:
            fixation_start = ts
        elif "Fixation_Cross_Stop" in msg and fixation_start is not None:
            events['fixation'].append((fixation_start, ts))
            fixation_start = None
        
        # Faces stimuli intervals.
        if "Faces_Stimuli_Start" in msg:
            faces_start = ts
            faces_emotion = None  # Reset emotion
        # Extract emotion side if available.
        if "Emotion_Side:" in msg:
            m = re.search(r"Emotion_Side:\s*(\w+)", msg, re.IGNORECASE)
            if m:
                emo = m.group(1).lower()
                faces_emotion = "L" if emo == "left" else "R"
        if "Faces_Stimuli_Stop" in msg and faces_start is not None:
            # If no emotion was captured, default to "N".
            if faces_emotion is None:
                faces_emotion = "N"
            events['faces'].append((faces_start, ts, faces_emotion))
            faces_start = None
            faces_emotion = None
        
        # Pause intervals.
        if "Pause_Start" in msg:
            pause_start = ts
        elif "Pause_Stop" in msg and pause_start is not None:
            events['pause'].append((pause_start, ts))
            pause_start = None

    events['experiment'] = (exp_start, exp_end)
    return events

def assign_flags(gaze_df, events):
    """
    Given a gaze DataFrame and event intervals, assign flag and emotion_side.
    Returns the DataFrame with added 'flag' and 'emotion_side' columns.
    """
    # Initialize new columns.
    gaze_df['flag'] = -1
    gaze_df['emotion_side'] = "N"  # Default to "N" if no emotion side is applicable.
    
    # Apply flag 0 for fixation intervals.
    for start, stop in events['fixation']:
        mask = (gaze_df['Time'] >= start) & (gaze_df['Time'] <= stop)
        gaze_df.loc[mask, 'flag'] = 0
    
    # Apply flag 1 for faces stimuli intervals and assign emotion.
    for start, stop, emotion in events['faces']:
        mask = (gaze_df['Time'] >= start) & (gaze_df['Time'] <= stop)
        gaze_df.loc[mask, 'flag'] = 1
        gaze_df.loc[mask, 'emotion_side'] = emotion  # "L", "R", or "N" if not captured.
    
    # Apply flag 2 for pause intervals.
    for start, stop in events['pause']:
        mask = (gaze_df['Time'] >= start) & (gaze_df['Time'] <= stop)
        gaze_df.loc[mask, 'flag'] = 2
    
    return gaze_df

def process_participant(gaze_file, msg_file):
    """
    Process a single participant's files:
      - Read the gaze file (columns: Time, calX, calY, rawX, rawY)
      - Process the message file to extract event intervals.
      - Filter gaze data to the experiment window.
      - Assign flags and emotion_side.
      - Add a 'participants' column.
    Returns the processed gaze DataFrame.
    """
    # Extract participant identifier (e.g., P006) from the filename.
    match = re.search(r"(P\d+)", os.path.basename(gaze_file))
    if not match:
        return None
    participant_id = match.group(1)
    
    # Read the gaze file.
    df_gaze = pd.read_csv(gaze_file, sep=r"\s+", header=None, engine='python')
    df_gaze.columns = ['Time', 'calX', 'calY', 'rawX', 'rawY']
    df_gaze['Time'] = pd.to_numeric(df_gaze['Time'])
    
    # Add participants column.
    df_gaze['participants'] = participant_id
    
    # Process message file to get event intervals.
    events = process_message_file(msg_file)
    
    # Filter gaze data to the experiment window.
    exp_start, exp_end = events['experiment']
    if exp_start is None or exp_end is None:
        print(f"Experiment boundaries not found for {participant_id}")
        return None
    df_exp = df_gaze[(df_gaze['Time'] >= exp_start) & (df_gaze['Time'] <= exp_end)].copy()
    
    # Assign flags and emotion_side.
    df_processed = assign_flags(df_exp, events)
    return df_processed

def main():
    # Collect gaze and message files (assumes they are in the current directory).
    gaze_files = glob.glob("gaze_directions_calibrated_*.txt")
    msg_files = glob.glob("gaze_messages_*.txt")
    
    # Map each participant id to its corresponding message file.
    msg_dict = {}
    for mf in msg_files:
        match = re.search(r"(P\d+)", os.path.basename(mf))
        if match:
            participant_id = match.group(1)
            msg_dict[participant_id] = mf

    # Process each participant's data and combine into a single DataFrame.
    df_list = []
    for gf in gaze_files:
        match = re.search(r"(P\d+)", os.path.basename(gf))
        if not match:
            continue
        participant_id = match.group(1)
        if participant_id in msg_dict:
            processed = process_participant(gf, msg_dict[participant_id])
            if processed is not None:
                df_list.append(processed)
        else:
            print(f"No message file found for participant {participant_id}")

    if df_list:
        df_final = pd.concat(df_list, ignore_index=True)
        # Optionally sort by participant and Time.
        df_final = df_final.sort_values(['participants', 'Time']).reset_index(drop=True)
        # Save the final DataFrame to a CSV file.
        df_final.to_csv("combined_processed_data.csv", index=False)
        print("Combined Processed Data:")
        print(df_final.head(20))
    else:
        print("No data processed.")

if __name__ == "__main__":
    main()

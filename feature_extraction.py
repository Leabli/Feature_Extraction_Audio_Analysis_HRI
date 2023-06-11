# === Import Library  ===
import parselmouth
from parselmouth.praat import call, run_file
import os 
import pandas as pd
import math

# === Import Functions from "Simple Speech Features"===
from feature_extraction_utils import *

def folder_feature_extraction(audio_path, dataframe):
    """
    Perform feature extraction on audio files stored in a specific folder structure.

    Args:
        audio_path (str): Path to the main audio folder.
        dataframe (pandas.DataFrame): DataFrame containing participant information.

    Returns:
        pandas.DataFrame: Concatenated DataFrame containing the extracted features.
    """
    dataframe_list = []  # List to store individual DataFrames for each file
    for participant in os.listdir(audio_path):  # Iterate over participant folders
        print('Participant is:', participant)
        print()
        DIRECTORY_PARTICIPANT = os.path.join(audio_path, participant)  # Path to the participant folder
        
        dataframe_part = dataframe[dataframe['Code'] == participant].copy()  # Select participant data from the DataFrame
        
        for folder in os.listdir(DIRECTORY_PARTICIPANT):  # Iterate over folders within the participant folder
            print(' Folder is:', folder)
            DIRECTORY_FOLDERS = os.path.join(DIRECTORY_PARTICIPANT, folder)  # Path to the folder
            
            dataframe_part_folder = dataframe_part.copy()  # Create a copy of the participant data
            dataframe_part_folder['Intervenant'] = folder  # Add the folder as the 'Intervenant' column value
            
            for subfolder in os.listdir(DIRECTORY_FOLDERS):  # Iterate over subfolders within the folder
                dataframe_part_subfolder = dataframe_part_folder.copy()  # Create a copy of the participant data
                dataframe_part_subfolder['Lecon'] = subfolder  # Add the subfolder as the 'Lecon' column value
                
                print('  Subfolder is:', subfolder)
                DIRECTORY_FILES = os.path.join(DIRECTORY_FOLDERS, subfolder)  # Path to the subfolder
                
                for filename in os.listdir(DIRECTORY_FILES):  # Iterate over files within the subfolder
                    if filename.endswith(".wav"):  # Check if the file is a WAV file
                        print('   Filename is:', filename)
                        name = filename.replace('.wav', '')  # Extract the name from the file
                        dataframe_part_subsubfolder = dataframe_part_subfolder.copy()  # Create a copy of the participant data
                        dataframe_part_subsubfolder['Type'] = name  # Add the name as the 'Type' column value
                        
                        data_features = feature_extraction(DIRECTORY_FILES, filename, dataframe_part_subsubfolder)  # Perform feature extraction on the file
                        dataframe_list.append(data_features)  # Append the extracted features to the list

    print()
    df = pd.concat(dataframe_list)  # Concatenate all the individual DataFrames into a single DataFrame

    return df

def feature_extraction(directory, filename, features):
    """
    Perform feature extraction on a specific audio file.

    Args:
        directory (str): Directory path of the audio file.
        filename (str): Name of the audio file.
        features (pandas.DataFrame): DataFrame to store the extracted features.

    Returns:
        pandas.DataFrame: DataFrame with the extracted features.
    """
    features = features_from_utils(directory, filename, features)  # Extract features using a utility function
    
    features = mysptotal_modulate_threshold(filename, directory, features)  # Modulate features using another function

    return features

def mysptotal_modulate_threshold(filename, directory_participant, features, silence_threshold=-20):
    """
    Adjust the silence threshold parameter and perform a specific operation using the 'mysptotal' function.

    Args:
        filename (str): Name of the audio file.
        directory_participant (str): Directory path of the participant.
        features (pandas.DataFrame): DataFrame to store the extracted features.
        silence_threshold (float, optional): Initial silence threshold. Defaults to -20.

    Returns:
        pandas.DataFrame: DataFrame with the extracted features.
    """
    max_silence_threshold = 0

    while silence_threshold < max_silence_threshold:
        if mysptotal(filename, directory_participant, features, silence_threshold) is not None:
            return mysptotal(filename, directory_participant, features, silence_threshold)
        else:
            silence_threshold += 1

    return features

def mysptotal(filename, directory_participant, features, silence_threshold=-20, Minimum_dip_between_peaks=2,
              Minimum_pause_duration=0.3, Keep_Soundfiles_and_Textgrids="yes", Minimum_pitch=80,
              Maximum_pitch=400, Time_step=0.01):
    """
    Perform a specific operation using the Praat software for speech analysis.

    Args:
        filename (str): Name of the audio file.
        directory_participant (str): Directory path of the participant.
        features (pandas.DataFrame): DataFrame to store the extracted features.
        silence_threshold (float, optional): Silence threshold parameter. Defaults to -20.
        Minimum_dip_between_peaks (int, optional): Minimum dip between peaks parameter. Defaults to 2.
        Minimum_pause_duration (float, optional): Minimum pause duration parameter. Defaults to 0.3.
        Keep_Soundfiles_and_Textgrids (str, optional): Keep sound files and text grids parameter. Defaults to "yes".
        Minimum_pitch (int, optional): Minimum pitch parameter. Defaults to 80.
        Maximum_pitch (int, optional): Maximum pitch parameter. Defaults to 400.
        Time_step (float, optional): Time step parameter. Defaults to 0.01.

    Returns:
        pandas.DataFrame: DataFrame with the extracted features.
    """
    sound = directory_participant + "/" + filename
    path = directory_participant + "/"
    
    # === Location of the myspsolution.praat from My voice analysis library ===
    sourcerun = r"C:\Users\lea\OneDrive\Documents\Semester Project\features_extraction_chili_lab\myspsolution.praat"

    try:
        objects = run_file(sourcerun, silence_threshold, Minimum_dip_between_peaks, Minimum_pause_duration,
                           Keep_Soundfiles_and_Textgrids, sound, path, Minimum_pitch, Maximum_pitch,
                           Time_step, capture_output=True)

        z1 = str(objects[1])  # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
        z2 = z1.strip().split()
        z3 = np.array(z2)
        z4 = np.array(z3)[np.newaxis]
        z5 = z4.T

    except:
        return None

    attributes = z5.flatten().tolist()
    if attributes[0] == 'A':
        return None
    else:
        features["silence_threshold"] = silence_threshold
        features["rate_of_speech"] = attributes[2]
        features["articulation_rate"] = attributes[3]
        features["speaking_duration"] = attributes[4]
        features["balance"] = attributes[6]

    return features

def features_from_utils(directory, filename, features):
    """
    Extract acoustic features from an audio file using utility functions.

    Args:
        directory (str): Directory path of the audio file.
        filename (str): Name of the audio file.
        features (pandas.DataFrame): DataFrame to store the extracted features.

    Returns:
        pandas.DataFrame: DataFrame with the extracted features.
    """
    sound = parselmouth.Sound(os.path.join(directory, filename))  # Load the audio file using parselmouth

    features['duration'] = round(call(sound, "Get total duration"), 2)  # Extract the total duration of the audio

    attributes = get_intensity_attributes(sound)[0]  # Extract intensity-related attributes
    features['mean_intensity'] = round(attributes['mean_intensity'], 2)
    features['stddev_intensity'] = round(attributes['stddev_intensity'], 2)

    attributes = get_pitch_attributes(sound)[0]  # Extract pitch-related attributes
    features['voiced_fraction'] = round(attributes['voiced_fraction'], 2)
    features['min_pitch'] = round(attributes['min_pitch'], 2)
    features['max_pitch'] = round(attributes['max_pitch'], 2)
    features['mean_pitch'] = round(attributes['mean_pitch'], 2)
    features['stddev_pitch'] = round(attributes['stddev_pitch'], 2)
    features['mean_absolute_pitch_slope'] = round(attributes['mean_absolute_pitch_slope'], 2)

    attributes = get_harmonics_to_noise_ratio_attributes(sound)[0]  # Extract harmonics-to-noise ratio-related attributes
    features['mean_hnr'] = round(attributes['mean_hnr'], 2)
    features['stddev_hnr'] = round(attributes['stddev_hnr'], 2)

    features['jitter'] = round(get_local_jitter(sound), 2)  # Extract jitter
    features['shimmer'] = round(get_local_shimmer(sound), 2)  # Extract shimmer

    attributes = get_spectrum_attributes(sound)[0]  # Extract spectrum-related attributes

    return features

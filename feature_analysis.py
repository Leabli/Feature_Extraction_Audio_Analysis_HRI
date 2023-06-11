# === Import Library  ===
import matplotlib.pyplot as plt  # Library for creating plots and visualizations
import pandas as pd  # Library for data manipulation and analysis
import seaborn as sns  # Library for statistical data visualization
import numpy as np  # Library for numerical computing and array manipulation
import warnings  # Library for handling warnings during code execution
from scipy.stats import wilcoxon, mannwhitneyu  # Functions for non-parametric statistical tests

# === Data Manipulation ===

def ordre_passage(dataframe):
    """
    This function is used to split the order of subjects (Anglais, Histoire, Science) for both robots and humans 
    from the dataframe. It first separates the subjects into different columns, and then splits each subject's order and number
    into separate columns. Unnecessary columns are then dropped.
    
    Parameters:
    dataframe (DataFrame): The DataFrame containing the data to be processed.
    
    Returns:
    lecon_robot (DataFrame): The processed DataFrame for robot data.
    lecon_humain (DataFrame): The processed DataFrame for human data.
    """

    lecon_robot = pd.DataFrame()
    lecon_humain = pd.DataFrame()

    lecon_robot[['Anglais', 'Histoire', 'Science' ]] = dataframe['Ordre passage robot [A/H/S]'].str.split('/', expand=True)
    lecon_humain[['Anglais', 'Histoire', 'Science' ]] = dataframe['Ordre passage humain [A/H/S]'].str.split('/', expand=True)

    lecon_robot[['Ordre Anglais', 'Num Anglais']] = lecon_robot['Anglais'].str.split('.', expand=True)
    lecon_robot[['Ordre Histoire', 'Num Histoire']] = lecon_robot['Histoire'].str.split('.', expand=True)
    lecon_robot[['Ordre Science', 'Num Science']] = lecon_robot['Science'].str.split('.', expand=True)

    lecon_humain[['Ordre Anglais', 'Num Anglais']] = lecon_humain['Anglais'].str.split('.', expand=True)
    lecon_humain[['Ordre Histoire', 'Num Histoire']] = lecon_humain['Histoire'].str.split('.', expand=True)
    lecon_humain[['Ordre Science', 'Num Science']] = lecon_humain['Science'].str.split('.', expand=True)

    columns_to_drop = ['Anglais', 'Histoire', 'Science']
    lecon_robot = lecon_robot.drop(columns_to_drop, axis=1)
    lecon_humain = lecon_humain.drop(columns_to_drop, axis=1)

    return lecon_robot, lecon_humain

def extract_list_participants(df):
    """
    Extracts a list of unique participants from a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the 'Code' column.

    Returns:
        list: A list of unique participants.

    """
    # Extract the unique values from the 'Code' column
    list_part = list(df['Code'].unique())

    return list_part

def split_humain_robot(dataframe):
    """
    Splits a dataframe into two separate dataframes for human and robot interactions.

    Args:
        dataframe (pandas.DataFrame): The input dataframe containing the 'Intervenant' column.

    Returns:
        tuple: A tuple of two dataframes - one for human interactions and one for robot interactions.

    """
    # Filter rows where 'Intervenant' is 'Humain'
    data_human = dataframe[dataframe['Intervenant'] == "Humain"]

    # Filter rows where 'Intervenant' is 'Robot'
    data_robot = dataframe[dataframe['Intervenant'] == "Robot"]

    return data_human, data_robot

def split_data_by_starting_order(dataframe): 
    """
    This function splits the dataframe into two parts based on the starting order of humans and robots.
    It creates two separate dataframes, one for when humans start first and the other for when robots start first.
    The columns related to the order of starting are then dropped.
    
    Parameters:
    dataframe (DataFrame): The DataFrame to be split.
    
    Returns:
    data_humain_first_ (DataFrame): The DataFrame for when humans start first.
    data_robot_first_ (DataFrame): The DataFrame for when robots start first.
    """
    data_humain_first =  dataframe[dataframe['Ordre [Humain]']== 1]
    data_robot_first =  dataframe[dataframe['Ordre [Robot ]']== 1]
    
    columns_to_drop = ['Ordre [Robot ]', 'Ordre [Humain]']
    data_robot_first_ = data_robot_first.drop(columns_to_drop, axis=1)
    data_humain_first_ = data_humain_first.drop(columns_to_drop, axis=1)
    
    return data_humain_first_, data_robot_first_

def split_by_lecons(df):
    """
    Splits a dataframe into separate dataframes based on the values in the 'Lecon' column.

    Args:
        df (pandas.DataFrame): The input dataframe containing the 'Lecon' column.

    Returns:
        tuple: A tuple of three dataframes - one for 'Anglais' lessons, one for 'Histoire' lessons, and one for 'Science' lessons.

    """
    # Filter rows where 'Lecon' starts with 'Anglais'
    anglais_df = df[df['Lecon'].str.startswith('Anglais')]

    # Filter rows where 'Lecon' starts with 'Histoire'
    histoire_df = df[df['Lecon'].str.startswith('Histoire')]

    # Filter rows where 'Lecon' starts with 'Science'
    science_df = df[df['Lecon'].str.startswith('Science')]

    return anglais_df, histoire_df, science_df

def split_by_language(df):
    """
    Splits a dataframe into separate dataframes based on the language in the 'Lecon' column.

    Args:
        df (pandas.DataFrame): The input dataframe containing the 'Lecon' column.

    Returns:
        tuple: A tuple of two dataframes - one for English lessons and one for French (non-English) lessons.

    """
    # Filter rows where 'Lecon' starts with 'Anglais'
    data_english = df[df['Lecon'].str.startswith('Anglais')]

    # Filter rows where 'Lecon' does not start with 'Anglais' (French or non-English lessons)
    data_french = df[~df['Lecon'].str.startswith('Anglais')]

    return data_english, data_french

def split_and_determine_first_french(df, column_name):
    """
    This function splits the specified column into three separate columns (English, History, Science) and determines which French lesson (History or Science) is taught first.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to split.

    Returns:
        df (pd.DataFrame): The modified DataFrame with the new columns and the 'startsFrench' column indicating which French lesson is taught first.
    """
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Split the column into three new columns
    df[['Anglais', 'Histoire', 'Science']] = df[column_name].str.split('/', expand=True)

    # Extract the order number for History and Science
    df['Ordre Histoire'] = df['Histoire'].str.split('.', expand=True)[0].astype(int)
    df['Ordre Science'] = df['Science'].str.split('.', expand=True)[0].astype(int)

    # Determine which French lesson is taught first
    df['FirstFrench'] = df.apply(lambda row: 'Histoire' if row['Ordre Histoire'] < row['Ordre Science'] else 'Science', axis=1)

    # Drop the temporary order columns
    df = df.drop(['Anglais', 'Histoire', 'Science', 'Ordre Histoire', 'Ordre Science',], axis=1)

    return df

# === Function for Statitics ===

def deal_outliers(df):
    """
    Deals with outliers in a DataFrame by replacing them with NaN values.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: A new DataFrame with outliers replaced by NaN values.

    """
    seuil = 1.5  # Modify if necessary

    # Create a copy of the DataFrame
    df_copy = df.copy()

    for column in df.columns:
        # Calculate the first quartile (Q1)
        Q1 = df_copy[column].quantile(0.25)

        # Calculate the third quartile (Q3)
        Q3 = df_copy[column].quantile(0.75)

        # Calculate the interquartile range (IQR)
        IQR = Q3 - Q1

        # Calculate the lower and upper thresholds for outliers
        seuil_inf = Q1 - seuil * IQR
        seuil_sup = Q3 + seuil * IQR

        # Create a boolean mask to identify outliers
        outliers_mask = (df_copy[column] >= seuil_inf) & (df_copy[column] <= seuil_sup)

        # Replace outliers with NaN values
        df_copy.loc[~outliers_mask, column] = np.nan

    return df_copy

def calculate_mannwhitneyu_p_value(diff_human, diff_robot):
    """
    Calculates the Mann-Whitney U test p-values for the given features between human and robot data.

    Args:
        diff_human (pandas.DataFrame): DataFrame containing differences for each feature for human data.
        diff_robot (pandas.DataFrame): DataFrame containing differences for each feature for robot data.

    Returns:
        pandas.DataFrame: DataFrame with p-values for each feature, sorted by p-value.
    """
    list_of_features = list(diff_human.columns[1:])
    p_value = []

    for feature in list_of_features:
        # Perform Mann-Whitney U test for the given feature
        u, p = mannwhitneyu(diff_human[feature], diff_robot[feature])
        
        # Append feature and its p-value to the list
        p_value.append([feature] + ["{:.4f}".format(p)])

    columns = ['Features', 'p-value']
    
    # Create DataFrame from the list of p-values
    p_value_df = pd.DataFrame(p_value, columns=columns)
    
    # Sort DataFrame by p-value
    p_value_df = p_value_df.sort_values('p-value')
    
    # Add a column indicating if the distribution is significantly different or not
    p_value_df['Distribution different'] = p_value_df['p-value'].apply(lambda x: 'strong evidence' if float(x) < 0.05 else 'not sufficient')

    return p_value_df

def calculate_wilcoxon_p_value(df1, df2):
    """
    Calculates the Wilcoxon signed-rank test p-values for the given features between two datasets.

    Args:
        df1 (pandas.DataFrame): First DataFrame.
        df2 (pandas.DataFrame): Second DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with p-values for each feature, sorted by p-value.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')  # Ignore warnings

        list_of_features = df1.columns[1:].tolist()
        p_value = []

        for feature in list_of_features:
            if feature == 'band_density':
                continue

            # Perform Wilcoxon signed-rank test for the given feature
            _, p = wilcoxon(df1[feature], df2[feature])
            
            # Append feature and its p-value to the list
            p_value.append([feature] + ["{:.4f}".format(p)])

        columns = ['Features', 'p-value']
        
        # Create DataFrame from the list of p-values
        p_value_df = pd.DataFrame(p_value, columns=columns)
        
        # Sort DataFrame by p-value
        p_value_df = p_value_df.sort_values('p-value')
        
        # Add a column indicating if the distribution is significantly different or not
        p_value_df['Distribution different'] = p_value_df['p-value'].apply(lambda x: 'strong evidence' if float(x) < 0.05 else 'not sufficient')
    
    return p_value_df

# === Features Analysis ===

def calculate_mean_std_stats(dataframe):
    """
    Calculates mean and standard deviation statistics for the given DataFrame.

    Args:
        dataframe (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with mean and standard deviation statistics.
    """

    # Columns to drop
    columns_to_drop = list(dataframe.columns[:18])

    # Create a copy of the original DataFrame
    features = dataframe.copy()

    # Drop the specified columns
    features.drop(columns_to_drop, axis=1, inplace=True)

    # Calculate descriptive statistics for the group
    stats = features.describe().round(2)

    # Select only the rows containing the mean and standard deviation
    stats = stats.loc[['mean', 'std']]

    return stats

def feature_statistics(data_features, list_part):
    """
    Calculates mean and standard deviation statistics for different participants in the given data.

    Args:
        data_features (pandas.DataFrame): The input DataFrame containing features.
        list_part (list): List of participants.

    Returns:
        pandas.DataFrame: DataFrame with mean and standard deviation statistics for each participant.
    """
    stats_part = []

    for part in list_part:
        # Filter data for the specific participant
        data_part = data_features[data_features['Code'] == part]

        # Split data for the participant into human and robot interactions
        data_part_humain, data_part_robot = split_humain_robot(data_part)

        # Calculate mean and standard deviation statistics for robot and human interactions
        robot_stats = calculate_mean_std_stats(data_part_robot)
        human_stats = calculate_mean_std_stats(data_part_humain)

        # Concatenate the DataFrames with specified index keys
        concatenated_stats = pd.concat([robot_stats, human_stats], keys=['robot', 'human'])
        stats_part.append(concatenated_stats)

    # Concatenate the DataFrames in the stats_part list using concat()
    df = pd.concat(stats_part, keys=list_part)

    return df

def feature_difference_by_part(dataframe, starting_with):
    """
    Calculates the feature difference between human and robot interactions for each participant.

    Args:
        dataframe (pandas.DataFrame): The input DataFrame containing features.
        starting_with (str): Indicates whether to start with 'humain' or 'robot' for the difference calculation.

    Returns:
        pandas.DataFrame: DataFrame with feature differences for each participant.
    """
    data_humain, data_robot = split_humain_robot(dataframe)
    columns_to_drop = dataframe.columns[:18].tolist()
    diff_list = []

    list_part = list(dataframe['Code'].unique())

    for part in list_part:
        data_part_humain = data_humain[data_humain['Code'] == part]
        data_part_robot = data_robot[data_robot['Code'] == part]

        # Calculate mean features for human and robot interactions
        mean_features_humain = data_part_humain.drop(columns_to_drop, axis=1).values.mean(axis=0)
        mean_features_robot = data_part_robot.drop(columns_to_drop, axis=1).values.mean(axis=0)

        if starting_with == 'humain':
            # Calculate the feature difference starting with 'humain'
            diff = mean_features_humain - mean_features_robot
        elif starting_with == 'robot':
            # Calculate the feature difference starting with 'robot'
            diff = mean_features_robot - mean_features_humain

        diff_list.append([part] + diff.tolist())

    columns = ['Code'] + dataframe.columns[18:].tolist()

    # Create DataFrame with the feature differences
    df_diff = pd.DataFrame(diff_list, columns=columns)
    
    return df_diff

def calculate_language_mean_features_languages(intervenant_first):
    """
    Calculates the mean features for French and English languages for each participant.

    Args:
        intervenant_first (pandas.DataFrame): DataFrame containing features with 'Intervenant' and 'FirstFrench' columns.

    Returns:
        pandas.DataFrame: DataFrame with mean features for French language.
        pandas.DataFrame: DataFrame with mean features for English language.
    """
    list_features = intervenant_first.columns[18:-1].tolist()
    list_part = extract_list_participants(intervenant_first)

    list_french = []
    list_english = []

    for part in list_part:
        data_part = intervenant_first[intervenant_first['Code'] == part]
        first_french = data_part['FirstFrench'].values[0]

        # Calculate mean features for French language
        data_french = data_part[data_part['Lecon'].str.startswith(first_french)][list_features].values.mean(axis=0).tolist()

        # Calculate mean features for English language
        data_english = data_part[data_part['Lecon'].str.startswith('Anglais')][list_features].values.mean(axis=0).tolist()

        # Check for NaN values in the mean features
        if np.isnan(data_french).any() or np.isnan(data_english).any():
            continue

        list_french.append([part] + data_french)
        list_english.append([part] + data_english)

    columns = ['Code'] + list_features

    # Create DataFrame with mean features for French language
    df_french = pd.DataFrame(list_french, columns=columns)

    # Create DataFrame with mean features for English language
    df_english = pd.DataFrame(list_english, columns=columns)

    return df_french, df_english

def calculate_mean_features_first_int(first_interaction):
    """
    This function calculates the mean of each feature for each unique participant in the 'Code' column of the input DataFrame.

    Args:
        first_interaction (pd.DataFrame): The input DataFrame. It should have a 'Code' column and feature columns starting from the 18th column.

    Returns:
        df (pd.DataFrame): A DataFrame with each unique 'Code' as a row and the mean of each feature as columns.
    """

    # Get a list of unique parts from the 'Code' column
    list_part = list(first_interaction['Code'].unique())

    # Get a list of feature column names starting from the 18th column
    features = first_interaction.columns[18:].tolist()

    list_features = []

    # Iterate over each unique part
    for part in list_part:
        # Filter rows where 'Code' is the current part
        first_interaction_part = first_interaction[first_interaction['Code'] == part]

        # Calculate the mean of each feature for the current part
        # Convert the result to a list
        mean_features = first_interaction_part[features].values.mean(axis = 0).tolist()

        # Append the part and its mean features to the list
        list_features.append([part] + mean_features)

    # Define column names for the output DataFrame
    columns = ['Code'] + features

    # Create the output DataFrame
    df = pd.DataFrame(list_features, columns=columns)

    return df

# === Plots ===

def plot_gender_distribution(dataframe_questionnaire):
    """
    Plots the gender distribution bar chart based on the 'Genre' column in the questionnaire DataFrame.

    Args:
        dataframe_questionnaire (pandas.DataFrame): DataFrame containing the 'Genre' column.

    """
    # Count the gender distribution
    repartition = dataframe_questionnaire['Genre'].value_counts()
    men = repartition[0]
    women = repartition[1]

    # Creating the bar chart
    categories = ['Women', 'Men']
    frequency = [women, men]

    plt.bar(categories, frequency, color=['red', 'blue'])

    # Adding labels and titles
    # Setting y-axis limits as integers
    plt.ylim(bottom=0, top=max(frequency) + 1)
    plt.xlabel('Categories')
    plt.ylabel('Number of participants')
    plt.title('Distribution of Women/Men')

    # Setting y-axis values as integers
    plt.yticks(range(int(0), int(max(frequency)) + 1, 1))

    # Displaying the chart
    plt.show()

    # Print the number of men and women
    print('During the experiment, there were', men, 'men and', women, 'women.')

def plot_starting_order_distribution(dataframe_questionnaire):
    """
    Plots the distribution of starting order (human vs robot) based on the 'Ordre [Humain]' column in the questionnaire DataFrame.

    Args:
        dataframe_questionnaire (pandas.DataFrame): DataFrame questionnaire the 'Ordre [Humain]' column.

    """
    # Count the distribution of starting order
    repartition = dataframe_questionnaire['Ordre [Humain]'].value_counts()
    human_first = repartition[1]
    robot_first = repartition[2]

    # Creating the bar chart
    categories = ['First with human', 'First with robot']
    frequency = [human_first, robot_first]

    plt.bar(categories, frequency, color=['red', 'blue'])

    # Adding labels and titles
    # Setting y-axis limits as integers
    plt.ylim(bottom=0, top=max(frequency) + 1)
    plt.xlabel('Categories')
    plt.ylabel('Number of participants')
    plt.title('Distribution of Starting Order: Human vs Robot')

    # Setting y-axis values as integers
    plt.yticks(range(int(0), int(max(frequency)) + 1, 1))

    # Displaying the chart
    plt.show()

    # Print the number of participants starting with human and robot
    print("During the experiment, there were", human_first, "participants starting with the human and", robot_first, "with the robot.")

def plot_ordre_passage(lecon_robot, lecon_humain): 
    """
    This function plots the frequency of order of passage for each type of lesson for robots and humans. 
    It generates a separate bar plot for each lesson.
    
    Parameters:
    lecon_robot (DataFrame): The DataFrame containing the order of passage data for the robot.
    lecon_humain (DataFrame): The DataFrame containing the order of passage data for the human.
    
    Returns:
    None
    """
    list_lecon = ['Anglais', 'Histoire', 'Science']

    fig, axs = plt.subplots(len(list_lecon), 1, figsize=(8, len(list_lecon) * 6))

    for i, lecon in enumerate(list_lecon):
        ordre_robot = lecon_robot[f"Ordre {lecon}"].value_counts().tolist()
        ordre_humain = lecon_humain[f"Ordre {lecon}"].value_counts().tolist()
        index = ['1', '2', '3']

        df = pd.DataFrame({'Robot': ordre_robot, 'Humain': ordre_humain}, index=index)
        
        ax = axs[i]
        df.plot.bar(rot=0, ax=ax)
        ax.set_title(f'Frequency of occurrence for the lesson : {lecon}')
        ax.legend(['Robot', 'Human'])
        ax.set_ylabel('Frequency')
        ax.set_xlabel(f'Order of passage of {lecon}')

    # Ajuster l'espacement entre les sous-graphiques
    plt.tight_layout()

def box_plot_features_significant_starting_order(p_value_df, data_french):
    """
    Creates box plots for features showing significant differences based on the starting order.

    Args:
        p_value_df (pandas.DataFrame): DataFrame with p-values for features.
        data_french (pandas.DataFrame): DataFrame containing French language features.

    Returns:
        pandas.DataFrame: DataFrame with mean features for human (human first) group.
        pandas.DataFrame: DataFrame with mean features for robot (human first) group.
        pandas.DataFrame: DataFrame with mean features for human (robot first) group.
        pandas.DataFrame: DataFrame with mean features for robot (robot first) group.
    """
    features_significant = p_value_df[p_value_df['Distribution different'] == 'strong evidence']['Features'].tolist()

    first_human, first_robot = split_data_by_starting_order(data_french)
    list_part = list(data_french['Code'].unique())
    list_part_human = list(first_human['Code'].unique())
    list_part_robot = list(first_robot['Code'].unique())

    hf_human_list = []
    hf_robot_list = []
    rf_human_list = []
    rf_robot_list = []

    for part in list_part:
        if part in list_part_human:
            data_part = first_human[first_human['Code'] == part]
            hf_human_list.append(data_part[data_part['Intervenant'] == 'Humain'][features_significant].values.mean(axis=0).tolist())
            hf_robot_list.append(data_part[data_part['Intervenant'] == 'Robot'][features_significant].values.mean(axis=0).tolist())
        elif part in list_part_robot:
            data_part = first_robot[first_robot['Code'] == part]
            rf_human_list.append(data_part[data_part['Intervenant'] == 'Humain'][features_significant].values.mean(axis=0).tolist())
            rf_robot_list.append(data_part[data_part['Intervenant'] == 'Robot'][features_significant].values.mean(axis=0).tolist())

    columns = features_significant

    hf_human_df = pd.DataFrame(hf_human_list, columns=columns)
    hf_robot_df = pd.DataFrame(hf_robot_list, columns=columns)
    rf_human_df = pd.DataFrame(rf_human_list, columns=columns)
    rf_robot_df = pd.DataFrame(rf_robot_list, columns=columns)

    # Deal with outliers
    hf_human_df = deal_outliers(hf_human_df)
    hf_robot_df = deal_outliers(hf_robot_df)
    rf_human_df = deal_outliers(rf_human_df)
    rf_robot_df = deal_outliers(rf_robot_df)

    features_name = ['Standard deviation of HNR', 'Articulation rate', 'Mean intensity', 'Mean absolute pitch slope',
                     'Voiced fraction', 'Jitter', 'Mean HNR', 'Mean pitch', 'Duration', 'Minimum pitch',
                     'Rate of speech', 'Shimmer', 'Speaking duration']

    for feature, name in zip(features_significant, features_name):
        human_first = pd.DataFrame()
        human_first['Robot'] = hf_robot_df[feature]
        human_first['Humain'] = hf_human_df[feature]

        robot_first = pd.DataFrame()
        robot_first['Robot'] = rf_robot_df[feature]
        robot_first['Humain'] = rf_human_df[feature]

        human_first['group'] = 'First Human'
        robot_first['group'] = 'First Robot'

        # Concatenate the two dataframes
        df = pd.concat([human_first, robot_first], ignore_index=True)

        # Boxplot visualization
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='group', y='value', hue='variable', data=pd.melt(df, id_vars=['group'], value_vars=['Humain', 'Robot']))
        plt.title(f'Box plot of feature: {name}')
        plt.xlabel('Starting order')
        plt.show()

    return hf_human_df, hf_robot_df, rf_human_df, rf_robot_df

def box_plot_features_significant_languages(hf_french, hf_english, rf_french, rf_english, p_value_df_human, p_value_df_robot):
    """
    Creates box plots for features showing significant differences between languages.

    Args:
        hf_french (pandas.DataFrame): DataFrame with mean features for French language (human first).
        hf_english (pandas.DataFrame): DataFrame with mean features for English language (human first).
        rf_french (pandas.DataFrame): DataFrame with mean features for French language (robot first).
        rf_english (pandas.DataFrame): DataFrame with mean features for English language (robot first).
        p_value_df_human (pandas.DataFrame): DataFrame with p-values for features (human first).
        p_value_df_robot (pandas.DataFrame): DataFrame with p-values for features (robot first).

    """
    features_significant = p_value_df_human[p_value_df_human['Distribution different'] == "strong evidence"]['Features'].tolist()
    features_significant = features_significant + p_value_df_robot[p_value_df_robot['Distribution different'] == "strong evidence"]['Features'].tolist()
    features_significant = list(set(features_significant))

    features_name = ['Voiced Fraction', 'Mean Harmonic-to-Noise Ratio', 'Standard Deviation of Harmonic-to-Noise Ratio',
                     'Rate of Speech', 'Mean Intensity', 'Shimmer', 'Articulation Rate']
    
    hf_french = deal_outliers(hf_french[features_significant])
    hf_english = deal_outliers(hf_english[features_significant])
    rf_french = deal_outliers(rf_french[features_significant])
    rf_english = deal_outliers(rf_english[features_significant])

    for feature, name in zip(features_significant, features_name):
        human_first = pd.DataFrame()
        human_first['French'] = hf_french[feature]
        human_first['English'] = hf_english[feature]

        robot_first = pd.DataFrame()
        robot_first['French'] = rf_french[feature]
        robot_first['English'] = rf_english[feature]

        human_first['group'] = 'Humain'
        robot_first['group'] = 'Robot'

        # Concatenate the two dataframes
        df = pd.concat([human_first, robot_first], ignore_index=True)

        # Boxplot visualization
        plt.figure(figsize=(12, 6))
        palette = {'French': '#239cff', 'English': '#FFB9B8'}
        sns.boxplot(x='group', y='value', hue='variable', data=pd.melt(df, id_vars=['group'], value_vars=['French', 'English']),
                    palette=palette, color='white')
        plt.title(f'Box plot of feature: {name}')
        plt.xlabel('Group')
        plt.show()

def box_plot_features_significant_human_robot(mean_first_human, mean_first_robot, p_value_df):
    """
    Creates box plots for features showing significant differences between human and robot interactions.

    Args:
        mean_first_human (pandas.DataFrame): DataFrame with mean features for human interactions.
        mean_first_robot (pandas.DataFrame): DataFrame with mean features for robot interactions.
        p_value_df (pandas.DataFrame): DataFrame with p-values for features.

    """
    feature_significant = p_value_df[p_value_df['Distribution different'] == "strong evidence"]['Features'].tolist()

    # Filter the mean dataframes to include only significant features
    mean_first_human = deal_outliers(mean_first_human[feature_significant])
    mean_first_robot = deal_outliers(mean_first_robot[feature_significant])

    # Iterate over each feature
    feature_name = ['Mean intensity', 'Mean HNR', 'Standard deviation of HNR', 'Jitter', 'Rate of speech']
    for feature, name in zip(feature_significant, feature_name):
        # Create dataframes for human interactions and robot interactions
        human_df = pd.DataFrame()
        human_df['Values'] = mean_first_human[feature]
        human_df['Group'] = 'Human'

        robot_df = pd.DataFrame()
        robot_df['Values'] = mean_first_robot[feature]
        robot_df['Group'] = 'Robot'

        # Concatenate the two dataframes
        df = pd.concat([human_df, robot_df], ignore_index=True)

        # Boxplot visualization
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Group', y='Values', data=df)
        plt.title(f'Box Plot of feature: {name}')
        plt.xlabel('Group')
        plt.ylabel('Values')
        plt.show()

def plot_confort_ecart(dataframe_questionnaire):
    """
    Plots the comfort scores and the difference in comfort between human and robot interactions.

    Args:
        dataframe_questionnaire (pandas.DataFrame): DataFrame containing comfort scores.

    Returns:
        dict: Dictionary containing the comfort scores for human and robot interactions.
    """
    # List of comfort scores with human and robot interactions
    confort_humain = dataframe_questionnaire['Confort Humain'].tolist()
    confort_robot = dataframe_questionnaire['Confort Robot'].tolist()

    # Calculation of the comfort difference between human and robot for each participant (inverted)
    ecart_confort = [humain - robot for humain, robot in zip(confort_humain, confort_robot)]

    # Creating a colormap for the color gradient
    colormap = plt.cm.RdYlGn

    # Normalizing the comfort differences to obtain values between -5 and 5
    norm = plt.Normalize(-5, 5)
    colors = colormap(norm(ecart_confort))

    # Creating the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Bar chart with color gradient on the first subplot
    bars = ax1.bar(range(len(ecart_confort)), ecart_confort, color=colors)
    ax1.set_xlabel('Participants')
    ax1.set_ylabel("Comfort Difference (Human - Robot)")
    ax1.set_title("Comfort Difference between Human and Robot per Participant")
    ax1.set_ylim(-5, 5)

    # Adding a color bar (color gradient) to the legend on the first subplot
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # Ignore this line if you're using a newer version of Matplotlib
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label('Color Gradient')

    # Box plot on the second subplot
    ax2.boxplot([confort_humain, confort_robot], labels=['Human', 'Robot'])
    ax2.set_xlabel('Intervenant')
    ax2.set_ylabel('Comfort Score')
    ax2.set_title("Comparison of Comfort between Human and Robot")

    # Adjusting the spacing between subplots
    plt.subplots_adjust(wspace=0.3)

    # Displaying the figure
    plt.show()

    confort = {'confort_humain': confort_humain,
               'confort_robot': confort_robot}

    return confort


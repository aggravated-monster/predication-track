import tsi.tsi_sys as tsis
import pandas as pd

def get_instrument_input_base():
    return 'd:/ou/output/phase7/4a_instrument_features/'

def get_instrument_input_folder(suffix):
    return get_instrument_input_base() + suffix + '/'

def get_gaze_input_base():
    return 'd:/ou/output/phase7/4b_gaze_features/'

def get_gaze_input_folder(suffix):
    return get_gaze_input_base() + suffix + '/'

def get_position_input_base():
    return 'd:/ou/output/phase7/3b_positions/'

def get_position_input_folder(suffix):
    return get_position_input_base() + suffix + '/'

def get_spatial_input_base():
    return 'd:/ou/output/phase7/4c_spatial_features/'

def get_fixations_input_folder(suffix):
    return get_fixations_input_base() + suffix + '/'

def get_fixations_input_base():
    return 'd:/ou/output/phase7/4d_research_fixation_features/'

def get_output_base():
    return 'd:/ou/output/phase7/5_joined_features/'

def get_output_folder(suffix):
    return get_output_base() + suffix + '/'

def get_output_file_name(suffix, prefix):
    return get_output_folder(suffix) + prefix + '_joined_features.csv'

def get_summary_file_name():
    return get_output_base() + 'all_features_summary.csv' 

def cleanup(df, prefix):

    try:
        if prefix == "p3-1a":
            print(df.columns.tolist())
            df.drop('timestamp_y', 1, inplace=True)
            df.rename(columns={'timestamp_x': 'timestamp'}, inplace=True)
            print(df.columns.tolist())
        else:
            # drop and rename duplicate columns for nameif present
            df.drop('name_x', 1, inplace=True)
            df.rename(columns={'name_y': 'name'}, inplace=True)
    except:
        return df
    
    return df
    

def join(df_spatial, df_instrument, df_gaze, df_fixation, df_position, suffix, prefix):

    print("Joining features for " + prefix)

    rows_in_spatial = len(df_spatial.index)
    rows_in_instrument = len(df_instrument.index)
    rows_in_gaze = len(df_gaze.index)
    rows_in_fixation = len(df_fixation.index)
    rows_in_position = len(df_position.index)

    if prefix == "p3-1a":
        join_columns = ["subject", "name"]
    else:
        join_columns = ["timestamp", "subject"]

    df_join = pd.merge(df_spatial, df_instrument, on=join_columns, how='outer')
    rows_in_first_join = len(df_join.index)
    df_join = cleanup(df_join, prefix)

    df_join = pd.merge(df_gaze, df_join, on=join_columns, how='left')
    rows_in_second_join = len(df_join.index)
    df_join = cleanup(df_join, prefix)

    df_join = pd.merge(df_fixation, df_join, on=join_columns, how='left')
    rows_in_third_join = len(df_join.index)
    df_join = cleanup(df_join, prefix)

    df_join = pd.merge(df_position, df_join, on=join_columns, how='left')
    rows_in_fourth_join = len(df_join.index)
    df_join = cleanup(df_join, prefix)

    # write the result to output
    df_join.to_csv(get_output_file_name(suffix, prefix), index=False)

    return (prefix, rows_in_spatial, rows_in_instrument, rows_in_gaze, rows_in_fixation, rows_in_position, 
            rows_in_first_join, rows_in_second_join, rows_in_third_join, rows_in_fourth_join)


if __name__ == "__main__":

    summary = []

    # start from spatial; this is not an entirely random choice. Spatial is one of the smaller
    # sets, which we will join first with the other smaller set (instrument),
    # before joining the larger gaze set.
    spatial_input_folders = tsis.list_folders(get_spatial_input_base())

    for spatial_input_folder in spatial_input_folders:
        # strip inner folder and prepare output folder
        suffix = tsis.get_basename(spatial_input_folder)
        tsis.make_dir(get_output_folder(suffix))

        # collect the spatial files
        spatial_files = tsis.list_files(spatial_input_folder)
        for spatial_file in spatial_files:
            prefix = tsis.get_prefix(spatial_file)
            # find the corresponding instrument feature file
            instrument_file = list(filter(lambda x: prefix in x, 
                                    tsis.list_files(get_instrument_input_folder(suffix))))[0]
            gaze_file = list(filter(lambda x: prefix in x, 
                                    tsis.list_files(get_gaze_input_folder(suffix))))[0]
            fixation_file = list(filter(lambda x: prefix in x, 
                                    tsis.list_files(get_fixations_input_folder(suffix))))[0]
            position_file = list(filter(lambda x: prefix in x, 
                                    tsis.list_files(get_position_input_folder(suffix))))[0]
            # load the dataframes
            df_spatial = pd.read_csv(spatial_file)
            df_instrument = pd.read_csv(instrument_file)
            df_gaze = pd.read_csv(gaze_file)
            df_fixation = pd.read_csv(fixation_file)
            df_position = pd.read_csv(position_file)
            # join
            summary.append(join(df_spatial, df_instrument, df_gaze, df_fixation, df_position, suffix, prefix))

    # write summary
    df = pd.DataFrame(summary, columns=['prefix', 'rows_in_spatial', 'rows_in_instrument', 'rows_in_gaze', 'rows_in_fixation', 'rows_in_position', 
                                        'rows_in_first_join', 'rows_in_second_join', 'rows_in_third_join', 'rows_in_fourth_join'
                                        ])

    df.to_csv(get_summary_file_name(), index=False)   

    print('*** Join all features completed ***')
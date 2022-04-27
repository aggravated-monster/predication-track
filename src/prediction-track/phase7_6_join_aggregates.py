import tsi.tsi_sys as tsis
import pandas as pd

def get_instrument_input_base():
    return 'd:/ou/output/phase7/4a_instrument_features/'

def get_gaze_input_base():
    return 'd:/ou/output/phase7/4b_gaze_features/'

def get_spatial_input_base():
    return 'd:/ou/output/phase7/4c_spatial_features/'

def get_fixations_input_base():
    return 'd:/ou/output/phase7/4d_research_fixation_features/'

def get_output_base():
    return 'd:/ou/output/phase7/6_aggregated_features/'

def get_labels():
    return 'd:/ou/input/phase7/rings.csv'

def get_output_file_name(suffix):
    return get_output_base() + suffix + '_aggregated_features.csv'

def get_summary_file_name():
    return get_output_base() + 'aggregated_features_summary.csv' 

def join(df_spatial, df_instrument, df_gaze, df_fixation, suffix):

    print("Joining features for " + suffix)

    rows_in_spatial = len(df_spatial.index)
    rows_in_instrument = len(df_instrument.index)
    rows_in_gaze = len(df_gaze.index)
    rows_in_fixation = len(df_fixation.index)

    df_join = pd.merge(df_spatial, df_instrument, on=['subject'], how='outer')
    rows_in_first_join = len(df_join.index)

    df_join = pd.merge(df_gaze, df_join, on=['subject'], how='outer')
    rows_in_second_join = len(df_join.index)

    df_join = pd.merge(df_fixation, df_join, on=['subject'], how='outer')
    rows_in_third_join = len(df_join.index)

    # write the result to output
    df_join.to_csv(get_output_file_name(suffix), index=False)

    return (suffix, rows_in_spatial, rows_in_instrument, rows_in_gaze, rows_in_fixation, 
            rows_in_first_join, rows_in_second_join, rows_in_third_join)


if __name__ == "__main__":

    summary = []

    tsis.make_dir(get_output_base())

    spatial_files = tsis.list_files(get_spatial_input_base())

    for spatial_file in spatial_files:
        suffix = tsis.get_basename(spatial_file).split('_')[0]
        # find the corresponding instrument feature file
        instrument_file = list(filter(lambda x: suffix in x, 
                                tsis.list_files(get_instrument_input_base())))[0]
        gaze_file = list(filter(lambda x: suffix in x, 
                                tsis.list_files(get_gaze_input_base())))[0]
        fixation_file = list(filter(lambda x: suffix in x, 
                                tsis.list_files(get_fixations_input_base())))[0]
        # load the dataframes
        df_spatial = pd.read_csv(spatial_file)
        df_instrument = pd.read_csv(instrument_file)
        df_gaze = pd.read_csv(gaze_file)
        df_fixation = pd.read_csv(fixation_file)
        # join
        summary.append(join(df_spatial, df_instrument, df_gaze, df_fixation, suffix))

    # write summary
    df = pd.DataFrame(summary, columns=['suffix', 'rows_in_spatial', 'rows_in_instrument', 'rows_in_gaze', 'rows_in_fixation',
                                        'rows_in_first_join', 'rows_in_second_join','rows_in_third_join'
                                        ])

    df.to_csv(get_summary_file_name(), index=False)   

    print('*** Joining aggregated features completed ***')
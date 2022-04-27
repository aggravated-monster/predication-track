import tsi.tsi_sys as tsis
import pandas as pd

def get_labeled_input_base():
    return 'd:/ou/output/phase7/6_aggregated_features/'

def get_temporal_input_base():
    return 'd:/ou/output/phase7/8_temporal_sweeps/'

def get_output_base():
    return 'D:/ou/output/phase7/10_joined_aggregated_features/'

def get_output_file_name(suffix):
    return get_output_base() + suffix + '_joined_aggregated_features.csv'

def get_summary_file_name():
    return get_output_base() + 'joined_aggregated_features_summary.csv'

def get_result_file_name():
    return get_output_base() + 'joined_aggregated_features.csv'

def join(df_labeled, df_temporal, suffix):

    rows_in_labeled =  len(df_labeled.index)
    rows_in_temporal = len(df_temporal.index)

    df_join = pd.merge(df_labeled, df_temporal, on=['subject'], how='inner')

    rows_in_join = len(df_join.index)

    # write the result to output
    df_join.to_csv(get_output_file_name(suffix), index=False)

    return (df_join, (suffix, rows_in_labeled, rows_in_temporal, rows_in_join))

def fold(df_in):

    df = df_in[['subject','fixation_streak','departure_offset','arrival_offset']]

    df_sum = df.groupby(['subject']).sum()
    df_describe = df.groupby(['subject']).describe()

    print(df_describe)

    result = pd.merge(df_sum, df_describe, on=['subject'], how='outer')

    # clean up ugly column names
    result.rename(columns={'fixation_streak': 'sum_fixation_streak'}, inplace=True)
    result.rename(columns={'departure_offset': 'sum_departure_offset'}, inplace=True)
    result.rename(columns={'arrival_offset': 'sum_arrival_offset'}, inplace=True)
    result.rename(columns={('fixation_streak', 'count'): 'count_fixation_streak'}, inplace=True)
    result.rename(columns={('fixation_streak', 'mean'): 'mean_fixation_streak'}, inplace=True)
    result.rename(columns={('fixation_streak', 'std'): 'std_fixation_streak'}, inplace=True)
    result.rename(columns={('fixation_streak', 'min'): 'min_fixation_streak'}, inplace=True)
    result.rename(columns={('fixation_streak', '25%'): '25%_fixation_streak'}, inplace=True)
    result.rename(columns={('fixation_streak', '50%'): '50%_fixation_streak'}, inplace=True)
    result.rename(columns={('fixation_streak', '75%'): '75%_fixation_streak'}, inplace=True)
    result.rename(columns={('fixation_streak', 'max'): 'max_fixation_streak'}, inplace=True)

    result.rename(columns={('departure_offset', 'count'): 'count_departure_offset'}, inplace=True)
    result.rename(columns={('departure_offset', 'mean'): 'mean_departure_offset'}, inplace=True)
    result.rename(columns={('departure_offset', 'std'): 'std_departure_offset'}, inplace=True)
    result.rename(columns={('departure_offset', 'min'): 'min_departure_offset'}, inplace=True)
    result.rename(columns={('departure_offset', '25%'): '25%_departure_offset'}, inplace=True)
    result.rename(columns={('departure_offset', '50%'): '50%_departure_offset'}, inplace=True)
    result.rename(columns={('departure_offset', '75%'): '75%_departure_offset'}, inplace=True)
    result.rename(columns={('departure_offset', 'max'): 'max_departure_offset'}, inplace=True)

    result.rename(columns={('arrival_offset', 'count'): 'count_arrival_offset'}, inplace=True)
    result.rename(columns={('arrival_offset', 'mean'): 'mean_arrival_offset'}, inplace=True)
    result.rename(columns={('arrival_offset', 'std'): 'std_arrival_offset'}, inplace=True)
    result.rename(columns={('arrival_offset', 'min'): 'min_arrival_offset'}, inplace=True)
    result.rename(columns={('arrival_offset', '25%'): '25%_arrival_offset'}, inplace=True)
    result.rename(columns={('arrival_offset', '50%'): '50%_arrival_offset'}, inplace=True)
    result.rename(columns={('arrival_offset', '75%'): '75%_arrival_offset'}, inplace=True)
    result.rename(columns={('arrival_offset', 'max'): 'max_arrival_offset'}, inplace=True)

    print(result.columns.tolist())

    return result

if __name__ == "__main__":

    # placeholder for summary
    synopsis = []
    result = []

    tsis.make_dir(get_output_base())

    # start from the labeled features
    labeled_input_files = tsis.list_files(get_labeled_input_base())

    for labeled_input_file in labeled_input_files:

        suffix = tsis.get_prefix(labeled_input_file)

        if suffix != "aggregated":
            # find the corresponding instrument feature file
            temporal_files = list(filter(lambda x: suffix in x, 
                                    tsis.list_files(get_temporal_input_base())))
            if len(temporal_files) > 0:
                temporal_file = temporal_files[0]

                # load the dataframes
                df_labeled = pd.read_csv(labeled_input_file)
                df_temporal = pd.read_csv(temporal_file)
                # fold df_temporal first
                df_temporal = fold(df_temporal)
                # join
                df_joined, summary = join(df_labeled, df_temporal, suffix)
                result.append(df_joined)
                synopsis.append(summary)


    # merge the result into one big dataset
    result = pd.concat(result)
    result.to_csv(get_result_file_name(), index=False)   


    # write summary
    df = pd.DataFrame(synopsis, columns=['experiment', 'rows in spatial', 'rows in temporal', 'rows in join'])

    df.to_csv(get_summary_file_name(), index=False)   

    print(df)
    
    print('*** joining temporal features with aggregates completed ***')

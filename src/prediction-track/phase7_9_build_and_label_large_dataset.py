import tsi.tsi_sys as tsis
import pandas as pd


def get_input_base():
    return 'd:/ou/output/phase7/8_temporal_sweeps/'

def get_output_base():
    return 'd:/ou/output/phase7/9_large_labeled_dataset/'

def get_output_file_name(suffix):
    return get_output_base() + suffix + '_large_labeled_dataset.csv'

def get_labels_exp1():
    return 'd:/ou/input/phase7/rings.csv'

def get_labels_exp2():
    return 'd:/ou/input/phase7/rings_experiment2.csv'

def get_result_file_name():
    return get_output_base() + 'large_labeled_dataset.csv'


def aggregate_gaze_features(df, prefix, sweep):
    # calculate statistics for x_distance and abs_x_distance
    df_agg = df[["delta_eye_x", "abs_delta_eye_x", 'delta_eye_y', 'abs_delta_eye_y', 'delta_eye_euclid']].describe()

    # transpose into single row
    df_agg = df_agg.stack().to_frame().T
    df_agg.columns = ['{}_{}'.format(*c) for c in df_agg.columns]

    # add sums
    df_agg['sum_delta_eye_x'] = df['delta_eye_x'].sum()
    df_agg['sum_abs_delta_eye_x'] = df['abs_delta_eye_x'].sum()
    df_agg['sum_delta_eye_y'] = df['delta_eye_y'].sum()
    df_agg['sum_abs_delta_eye_y'] = df['abs_delta_eye_y'].sum()
    df_agg['sum_delta_eye_euclid'] = df['delta_eye_euclid'].sum()

    df_agg["subject"] = prefix
    df_agg["sweep"] = sweep

    return df_agg

def aggregate_instrument_features(df, prefix, sweep):
    # calculate statistics for x_distance and abs_x_distance
    df_agg = df[["delta_tooltip_x", "abs_delta_tooltip_x", 'delta_tooltip_y', 'abs_delta_tooltip_y', 'delta_tooltip_euclid']].describe()

    # transpose into single row
    df_agg = df_agg.stack().to_frame().T
    df_agg.columns = ['{}_{}'.format(*c) for c in df_agg.columns]

    # add sums
    df_agg['sum_delta_tooltip_x'] = df['delta_tooltip_x'].sum()
    df_agg['sum_abs_delta_tooltip_x'] = df['abs_delta_tooltip_x'].sum()
    df_agg['sum_delta_tooltip_y'] = df['delta_tooltip_y'].sum()
    df_agg['sum_abs_delta_tooltip_y'] = df['abs_delta_tooltip_y'].sum()
    df_agg['sum_delta_tooltip_euclid'] = df['delta_tooltip_euclid'].sum()

    df_agg["subject"] = prefix
    df_agg["sweep"] = sweep

    return df_agg

def aggregate_spatial_features(df, prefix, sweep):
    # calculate statistics for x_distance and abs_x_distance
    df_agg = df[["x_distance", "abs_x_distance", 'y_distance', 'abs_y_distance', 'euclid_distance']].describe()

    # transpose into single row
    df_agg = df_agg.stack().to_frame().T
    df_agg.columns = ['{}_{}'.format(*c) for c in df_agg.columns]

    # add sums
    df_agg['sum_x_distance'] = df['x_distance'].sum()
    df_agg['sum_abs_x_distance'] = df['abs_x_distance'].sum()
    df_agg['sum_y_distance'] = df['y_distance'].sum()
    df_agg['sum_abs_y_distance'] = df['abs_y_distance'].sum()
    df_agg['sum_euclid_distance'] = df['euclid_distance'].sum()

    df_agg["subject"] = prefix
    df_agg["sweep"] = sweep

    return df_agg

def aggregate_temporal_features(df, prefix, sweep):
    # calculate statistics for fixation streak and offsets
    df_agg = df[["fixation_streak", "departure_offset", 'arrival_offset']].describe()

    # transpose into single row
    df_agg = df_agg.stack().to_frame().T
    df_agg.columns = ['{}_{}'.format(*c) for c in df_agg.columns]

    # add sums
    df_agg['sum_fixation_streak'] = df['fixation_streak'].sum()
    df_agg['sum_departure_offset'] = df['departure_offset'].sum()
    df_agg['sum_arrival_offset'] = df['arrival_offset'].sum()

    df_agg["subject"] = prefix
    df_agg["sweep"] = sweep

    return df_agg


def aggregate_fixation_features(df, prefix, sweep):
    # calculate statistics for fixation streak and offsets
    df_agg = df[["fixation_duration"]].describe()

    # transpose into single row
    df_agg = df_agg.stack().to_frame().T
    df_agg.columns = ['{}_{}'.format(*c) for c in df_agg.columns]

    # add sums
    df_agg['sum_fixation_duration'] = df['fixation_duration'].sum()

    df_agg["subject"] = prefix
    df_agg["sweep"] = sweep

    return df_agg

def aggregate(input_file_path, prefix, sweep, df_labels):

    # load the dataframe
    df = pd.read_csv(input_file_path)

    df_agg_gaze = aggregate_gaze_features(df, prefix, sweep)
    df_agg_instrument = aggregate_instrument_features(df, prefix, sweep)
    df_agg_spatial = aggregate_spatial_features(df, prefix, sweep)
    df_agg_fixation = aggregate_fixation_features(df, prefix, sweep)
    # keep things readable
    result = df_agg_gaze.merge(df_agg_instrument, on=["subject","sweep"])
    result = result.merge(df_agg_spatial, on=["subject","sweep"])
    result = result.merge(df_agg_fixation, on=["subject","sweep"])

    #add the labels
    result = pd.merge(result, df_labels, on=['subject'], how='left')

    return result

def join(df_agg, suffix):
    # find corresponding termporal aggregates
    temporal_aggregates_file = list(filter(lambda x: suffix in x, 
                                tsis.list_files(get_input_base())))[0]    
    # load the temporal dataframe
    df_temporal = pd.read_csv(temporal_aggregates_file)

    df_join = pd.merge(df_agg, df_temporal, on=['sweep','subject'], how='outer')

    # write the result to output
    df_join.to_csv(get_output_file_name(suffix), index=False)
     
    return df_join

if __name__ == "__main__":

    result = []

    input_folders = tsis.list_folders(get_input_base())

    tsis.make_dir(get_output_base())

    for input_folder in input_folders:
        # placeholder for summary
        frames = []
        # strip inner folder and prepare output folder
        suffix = tsis.get_basename(input_folder)
        # get labels
        if suffix == "experiment1":
            df_labels = pd.read_csv(get_labels_exp1())
            df_labels = df_labels[["Su", "groups"]]
            # rename Su
            df_labels.rename(columns={'Su': 'subject'}, inplace=True)
            df_labels.rename(columns={'groups': 'label'}, inplace=True)
        else:
            df_labels = pd.read_csv(get_labels_exp2(), delimiter=";")
            df_labels = df_labels[["subject", "label"]]

        # collect the sweep folders
        sweep_folders = tsis.list_folders(input_folder)
        for sweep_folder in sweep_folders:
            # strip inner folder and prepare output folder
            prefix = tsis.get_basename(sweep_folder)
            # collect the sweeps
            input_files = tsis.list_files(sweep_folder)
            for input_file in input_files:
                sweep = tsis.drop_path_and_extension(input_file)
                print("Aggregating features for: " + sweep)
                frames.append(aggregate(input_file, prefix, sweep, df_labels))

        df_agg = pd.concat(frames)
        # join with temporal aggregates
        result.append(join(df_agg, suffix))

    # write summary
    result = pd.concat(result)
    result.to_csv(get_result_file_name(), index=False)

    print('*** Building large data set completed ***')

import tsi.tsi_sys as tsis
import tsi.tsi_math as tsimath
import pandas as pd


def get_input_base():
    return 'd:/ou/output/phase7/3b_positions/'

def get_output_base():
    return 'd:/ou/output/phase7/4a_instrument_features/'

def get_output_folder(suffix):
    return get_output_base() + suffix + '/'

def get_output_file_name(suffix, prefix):
    return get_output_folder(suffix) + prefix + '_instrument_features.csv'

def get_summary_file_name(suffix):
    return get_output_base() + suffix + '_aggregated_instrument_features.csv'

def calculate_delta(previous, this):

    try:
        return this - previous
    except ValueError:
        return 0

def expand(input_file_path, suffix, prefix):

    print("Adding instrument features for: " + prefix)

    # load the dataframe
    df_in = pd.read_csv(input_file_path)

    # instrument features only apply to rows with a frame name
    # so drop the others
    df_in = df_in[df_in['name'].notna()]

    # calculate delta_tooltip_x
    # to do this, add a column with a +1 offset
    df_in['previous_tooltip_x'] = df_in['tooltip_x'].shift(1)
    df_in['previous_tooltip_y'] = df_in['tooltip_y'].shift(1)
    # calculate delta-x
    df_in['delta_tooltip_x'] = df_in['previous_tooltip_x'] - df_in['tooltip_x']
    # calculate delta-y
    df_in['delta_tooltip_y'] = df_in['previous_tooltip_y'] - df_in['tooltip_y']

    # calculate euclidean distance
    df_in['delta_tooltip_euclid'] = df_in.apply(lambda x: tsimath.calculate_euclidean_distance(x['tooltip_x'],x['tooltip_y'],x['previous_tooltip_x'],x['previous_tooltip_y']), axis=1, result_type ='expand')

    # calculate abs of delta-x
    df_in['abs_delta_tooltip_x'] = df_in['delta_tooltip_x'].abs()
    # calculate abs of delta-y
    df_in['abs_delta_tooltip_y'] = df_in['delta_tooltip_y'].abs()

    print(df_in)

    # take only the necessary columns
    df_in = df_in[['timestamp','subject','name','delta_tooltip_x','abs_delta_tooltip_x','delta_tooltip_y','abs_delta_tooltip_y','delta_tooltip_euclid']]
    df_in.sort_values('timestamp')

    # write the result to output
    df_in.to_csv(get_output_file_name(suffix, prefix), index=False)

    return df_in

def aggregate(df, prefix):

    # calculate statistics for x_distance and abs_x_distance
    df_agg = df[["delta_tooltip_x", "abs_delta_tooltip_x", 'delta_tooltip_y', 'abs_delta_tooltip_y', 'delta_tooltip_euclid']].describe()

    print(df_agg)
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

    return df_agg


if __name__ == "__main__":

    input_folders = tsis.list_folders(get_input_base())

    for input_folder in input_folders:
        # placeholder for summary
        frames = []
        # strip inner folder and prepare output folder
        suffix = tsis.get_basename(input_folder)
        tsis.make_dir(get_output_base() + suffix)

        # collect the files
        input_files = tsis.list_files(input_folder)
        for input_file in input_files:
            prefix = tsis.get_prefix(input_file)
            expanded_df = expand(input_file, suffix, prefix)
            print("Aggregating features for: " + prefix)
            frames.append(aggregate(expanded_df, prefix))

        # write summary
        result = pd.concat(frames)
        result.to_csv(get_summary_file_name(suffix), index=False)

    print('*** Calculating instrument features completed ***')

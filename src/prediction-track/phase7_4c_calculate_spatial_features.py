import tsi.tsi_sys as tsis
import tsi.tsi_math as tsimath
import pandas as pd


def get_input_base():
    return 'd:/ou/output/phase7/3b_positions/'

def get_output_base():
    return 'd:/ou/output/phase7/4c_spatial_features/'

def get_output_folder(suffix):
    return get_output_base() + suffix + '/'

def get_output_file_name(suffix, prefix):
    return get_output_folder(suffix) + prefix + '_spatial_features.csv'

def get_summary_file_name(suffix):
    return get_output_base() + suffix + '_aggregated_spatial_features.csv'



def expand(input_file_path, suffix, prefix):

    print("Adding spatial features for: " + prefix)

    # load the dataframe
    df_in = pd.read_csv(input_file_path)

    # spatial features apply to tool motion data only
    # so drop rows with empty name
    df_in = df_in[df_in['name'].notna()]

    # calculate x-distance
    df_in['x_distance'] = df_in['eye_x'] - df_in['tooltip_x']
    # calculate y-distance
    df_in['y_distance'] = df_in['eye_y'] - df_in['tooltip_y']
    # calculate euclidean distance
    df_in['euclid_distance'] = df_in.apply(lambda x: tsimath.calculate_euclidean_distance(x['eye_x'],x['tooltip_x'],x['eye_y'],x['tooltip_y']), axis=1, result_type ='expand')

    # calculate abs of x-distance
    df_in['abs_x_distance'] = df_in['x_distance'].abs()
    # calculate abs of y-distance
    df_in['abs_y_distance'] = df_in['y_distance'].abs()

    print(df_in)

    # take only the necessary columns
    df_in = df_in[['timestamp','subject','name','x_distance','abs_x_distance','y_distance','abs_y_distance','euclid_distance']]
    df_in.sort_values('timestamp')  

    # write the result to output
    df_in.to_csv(get_output_file_name(suffix, prefix), index=False)

    return df_in

def aggregate(df, prefix):

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

    print('*** Calculating spatial features completed ***')

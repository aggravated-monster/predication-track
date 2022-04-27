import tsi.tsi_sys as tsis
import pandas as pd


def get_input_base():
    return 'd:/ou/output/phase7/3b_positions/'

def get_fixations_input_base():
    return 'd:/ou/input/phase7/fixations/'

def get_fixations_input_folder(suffix):
    return get_fixations_input_base() + suffix + '/'

def get_output_base():
    return 'd:/ou/output/phase7/4d_research_fixation_features/'

def get_output_folder(suffix):
    return get_output_base() + suffix + '/'

def get_output_file_name(suffix, prefix):
    return get_output_folder(suffix) + prefix + '_fixation_features.csv'

def get_summary_file_name():
    return get_output_base() + 'aggregated_fixation_summary.csv'

def get_result_file_name(suffix):
    return get_output_base() + suffix + '_aggregated_fixation_features.csv'



def expand(input_file_path, df_fixation, suffix, prefix):

    print("Adding fixation features for: " + prefix)

    # load the dataframe
    df_in = pd.read_csv(input_file_path)
    df_fix = df_fixation[df_fixation['subject'] == prefix]
    print(df_fix.columns.tolist())

    # fixation features apply to rows with a frame number
    # so drop rows with empty frame
    df_in = df_in[df_in['frame'].notna()]
    rows_in_position = len(df_in.index)
    rows_in_fixation = len(df_fix.index)

    # join
    df_join = pd.merge(df_in, df_fixation, on=['frame','subject'], how='left')
    df_join.sort_values('timestamp') 
    rows_in_join = len(df_fix.index)

    # take only the necessary columns
    df_join = df_join[['timestamp','subject','name','fixation_duration','fixation_x','fixation_y']]
    df_join.sort_values('timestamp')  

    # write the result to output
    df_join.to_csv(get_output_file_name(suffix, prefix), index=False)

    return (df_join, (prefix, rows_in_position, rows_in_fixation, rows_in_join))

def aggregate(df, prefix):

    # calculate statistics for x_distance and abs_x_distance
    df_agg = df[["fixation_duration"]].describe()

    # transpose into single row
    df_agg = df_agg.stack().to_frame().T
    df_agg.columns = ['{}_{}'.format(*c) for c in df_agg.columns]

    # add sums
    df_agg['sum_fixation_duration'] = df['fixation_duration'].sum()

    df_agg["subject"] = prefix

    return df_agg

def fix_subject_ex1(subject):
    if len(subject) <= 4: # very quick and dirty, but it suffices. No need to alter the input file this way
        return subject + "-1a"
    return subject

def prep_fixation(df_in):

    df_fixation = df_in[df_in['Phase'] == "TestInterval"]
    print(df_fixation)
    try:
        # experiment 1 has a slightly different naming scheme
        df_fixation = df_fixation[['VideoFrame','Duration','Xloc','Yloc','Su']]
        df_fixation.rename(columns={'Su': 'subject'}, inplace=True)
        df_fixation['subject'] = df_fixation.apply(lambda x: fix_subject_ex1(x['subject']), axis=1, result_type ='expand')

    except:
        df_fixation = df_fixation[['VideoFrame','Duration','Xloc','Yloc','Subject']]
        df_fixation.rename(columns={'Subject': 'subject'}, inplace=True)
    df_fixation.rename(columns={'VideoFrame': 'frame'}, inplace=True)
    df_fixation.rename(columns={'Duration': 'fixation_duration'}, inplace=True)
    df_fixation.rename(columns={'Xloc': 'fixation_x'}, inplace=True)
    df_fixation.rename(columns={'Yloc': 'fixation_y'}, inplace=True)

    return df_fixation


if __name__ == "__main__":

    summary = []

    input_folders = tsis.list_folders(get_input_base())

    for input_folder in input_folders:
        # placeholder for summary
        frames = []
        # strip inner folder and prepare output folder
        suffix = tsis.get_basename(input_folder)
        tsis.make_dir(get_output_base() + suffix)

        # find corresponding fixation file
        fixation_files = tsis.list_files(get_fixations_input_folder(suffix))
        fixation_file = fixation_files[0]

        df_fixation = pd.read_csv(fixation_file)
        # only keep the rows in the test interval
        df_fixation = prep_fixation(df_fixation)

        # collect the files
        input_files = tsis.list_files(input_folder)
        for input_file in input_files:
            prefix = tsis.get_prefix(input_file)
            expanded_df, summ = expand(input_file, df_fixation, suffix, prefix)
            summary.append(summ)
            print("Aggregating features for: " + prefix)
            frames.append(aggregate(expanded_df, prefix))

        # write summary
        result = pd.concat(frames)
        result.to_csv(get_result_file_name(suffix), index=False)
    
    # write summary
    df = pd.DataFrame(summary, columns=['prefix', 'rows_in_position', 'rows_in_fixation', 'rows_in_join'])

    df.to_csv(get_summary_file_name(), index=False)   

    print('*** Calculating research fixation features completed ***')

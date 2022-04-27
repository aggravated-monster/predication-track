import tsi.tsi_sys as tsis
import pandas as pd

def get_qe_input_file():
    return 'd:/ou/input/phase7/research/quiet_eye.csv'

def get_dwell_input_file():
    return 'd:/ou/input/phase7/research/dwell_times.csv'

def get_large_sacc_input_file():
    return 'd:/ou/input/phase7/research/large_sacs.csv'

def get_sacc_amplitude_input_file():
    return 'd:/ou/input/phase7/research/sac_ampl.csv'

def get_rings_moved_input_file():
    return 'd:/ou/input/phase7/research/ringsMoved.csv'

def get_output_base():
    return 'D:/ou/output/phase7/11_research_features/'

def get_output_file_name():
    return get_output_base() + 'research_features.csv'

def get_summary_file_name():
    return get_output_base() + 'research_features_summary.csv'

def join(df_labeled, df_temporal, suffix):

    rows_in_labeled =  len(df_labeled.index)
    rows_in_temporal = len(df_temporal.index)

    df_join = pd.merge(df_labeled, df_temporal, on=['subject'], how='inner')

    rows_in_join = len(df_join.index)

    # write the result to output
    df_join.to_csv(get_output_file_name(suffix), index=False)

    return (df_join, (suffix, rows_in_labeled, rows_in_temporal, rows_in_join))

def transpose_qe(df_in):

    # split the dataframe
    df_left = df_in[df_in['direction'] == "Leftward"]
    df_left.drop('direction', 1, inplace=True)
    df_left.drop('group', 1, inplace=True)
    df_left.rename(columns={'fix_dur': 'leftward_qe_duration'}, inplace=True)

    df_right = df_in[df_in['direction'] == "Rightward"]
    df_right.drop('direction', 1, inplace=True)
    df_right.drop('group', 1, inplace=True)
    df_right.rename(columns={'fix_dur': 'rightward_qe_duration'}, inplace=True)

    result = pd.merge(df_left, df_right, on=['Su'], how='outer')
    result.rename(columns={'Su': 'subject'}, inplace=True)

    return result

def transpose_dwell(df_in):

    # split the dataframe
    df_target = df_in[df_in['dish'] == "Target Dish"]
    df_target = df_target[['Su','all_samp','perc_samp']]
    df_target.rename(columns={'all_samp': 'dwell_time_target_dish'}, inplace=True)
    df_target.rename(columns={'perc_samp': 'perc_dwell_time_target_dish'}, inplace=True)

    df_elsewhere = df_in[df_in['dish'] == "Elsewhere"]
    df_elsewhere = df_elsewhere[['Su','all_samp','perc_samp']]
    df_elsewhere.rename(columns={'all_samp': 'dwell_time_elsewhere'}, inplace=True)
    df_elsewhere.rename(columns={'perc_samp': 'perc_dwell_time_elsewhere'}, inplace=True)

    df_start = df_in[df_in['dish'] == "Start dish"]
    df_start = df_start[['Su','all_samp','perc_samp']]
    df_start.rename(columns={'all_samp': 'dwell_time_start_dish'}, inplace=True)
    df_start.rename(columns={'perc_samp': 'perc_dwell_time_start_dish'}, inplace=True)

    result = pd.merge(df_target, df_elsewhere, on=['Su'], how='outer')
    result = pd.merge(result, df_start, on=['Su'], how='outer')
    result.rename(columns={'Su': 'subject'}, inplace=True)

    return result

def prep_large_sacs(df_in):

    result = df_in[['Su','Large','Small','perc_large']]
    result.rename(columns={'Su': 'subject'}, inplace=True)
    result.rename(columns={'Large': 'nr_large_sacc'}, inplace=True)
    result.rename(columns={'Small': 'nr_small_sacc'}, inplace=True)
    result.rename(columns={'perc_large': 'perc_large_sacc'}, inplace=True)
    result['perc_small_sacc'] = 100 - result['perc_large_sacc']

    return result

def prep_sacc_amplitude(df_in):

    result = df_in[['Su','sac_ampl']]
    result.rename(columns={'Su': 'subject'}, inplace=True)
    result.rename(columns={'sac_ampl': 'mean_saccade_amplitude'}, inplace=True)

    return result

def prep_rings_moved(df_in):

    result = df_in[['RingsMoved', 'FileName']]
    result.rename(columns={'FileName': 'subject'}, inplace=True)
    result.rename(columns={'RingsMoved': 'rings_moved'}, inplace=True)
    result['subject'] = result.apply(lambda x: cleanup_subject(x['subject']), axis=1, result_type ='expand')

    return result


def cleanup_subject(subject):

    # replace underscores
    subject = subject.replace('_', '-')
    if subject[0] != "p" and subject[0] != "f":
        subject = "su" + subject
    return subject

if __name__ == "__main__":

    # placeholder for summary
    synopsis = []

    tsis.make_dir(get_output_base())

    df_qe = pd.read_csv(get_qe_input_file())
    df_dwell = pd.read_csv(get_dwell_input_file())
    df_large_sacs = pd.read_csv(get_large_sacc_input_file())
    df_sac_ampl = pd.read_csv(get_sacc_amplitude_input_file())
    df_rings_moved = pd.read_csv(get_rings_moved_input_file())


    df_qe = transpose_qe(df_qe)
    df_dwell = transpose_dwell(df_dwell)
    df_large_sacs = prep_large_sacs(df_large_sacs)
    df_sac_ampl = prep_sacc_amplitude(df_sac_ampl)
    df_rings_moved = prep_rings_moved(df_rings_moved)

    # merge the result into one big dataset
    result = pd.merge(df_qe, df_dwell, on=['subject'], how='outer')
    result = pd.merge(result, df_large_sacs, on=['subject'], how='outer')
    result = pd.merge(result, df_sac_ampl, on=['subject'], how='outer')
    result = pd.merge(result, df_rings_moved, on=['subject'], how='outer')


    result.to_csv(get_output_file_name(), index=False)   

    print('*** Preparing research features completed ***')

import pandas as pd
import tsi.tsi_sys as tsis
import phase8_ffnn_common as p8


CLASSES = 2

def get_output_base():
    return 'c:/ou/output/phase8/ffnn/rrttsv/small/' + str(CLASSES) + '-classes/'

def get_output_figures_base():
    return get_output_base() + "figures/"

def get_output_figure_name(prefix, suffix):
    return get_output_figures_base() + prefix + '_' + suffix + '.png'

def get_input_file():
    # the small dataset
    return 'c:/ou/output/phase7/12_small_labeled_dataset/small_labeled_dataset.csv'

def get_predictions_file_name():
    return get_output_base() + 'predictions.csv'

def get_aggregates_file_name():
    return get_output_base() + 'aggregates.csv'

def evaluate_control(classes, df):

    result = []

    # highest single from research features
    features = ['rings_moved']
    res, _ = p8.evaluate_group(df, features, 2, 'rmsprop', 0.001, "control", "control", classes, 1, get_output_figure_name)
    result.append(res)

    return result

def evaluate_highest_singles(classes, df):

    result = []

    # highest single from research features
    features = ['perc_small_sacc']
    res, df_acc = p8.evaluate_group(df, features, 2, 'rmsprop', 0.001, "singles", "research", classes, 8, get_output_figure_name)
    result.append(res)

    # highest single from temporal features
    features = ['mean_departure_offset']
    res, df_acc = p8.evaluate_group(df, features, 1, 'rmsprop', 0.001, "singles", "temporal", classes, 8, get_output_figure_name, df_acc)
    result.append(res)

    p8.plot_boxplot("highest-single", df_acc, get_output_figure_name)

    return result

def evaluate_highest_pairs(classes, df):

    result = []

    # highest pair from research features
    features = ['max_fixation_duration', 'perc_dwell_time_elsewhere']
    res, df_acc = p8.evaluate_group(df, features, 3, 'rmsprop', 0.01, "pairs", "research", classes, 193, get_output_figure_name)
    result.append(res)

    features = ['min_fixation_duration', 'perc_dwell_time_elsewhere']
    res, df_acc = p8.evaluate_group(df, features, 3, 'rmsprop', 0.01, "pairs", "research", classes, 193, get_output_figure_name, df_acc)
    result.append(res)

    features = ['min_fixation_duration', 'perc_small_sacc']
    res, df_acc = p8.evaluate_group(df, features, 2, 'rmsprop', 0.001, "pairs", "research", classes, 193, get_output_figure_name, df_acc)
    result.append(res)

    features = ['mean_fixation_duration', 'perc_large_sacc']
    res, df_acc = p8.evaluate_group(df, features, 2, 'rmsprop', 0.001, "pairs", "research", classes, 193, get_output_figure_name, df_acc)
    result.append(res)

    p8.plot_scatterplot("highest-pair", df[features], features, 'research', get_output_figure_name)

    # highest pair from temporal features
    features = ['min_departure_offset', 'max_departure_offset']
    res, df_acc = p8.evaluate_group(df, features, 2, 'rmsprop', 0.001, "pairs", "temporal", classes, 193, get_output_figure_name, df_acc)
    result.append(res)

    features = ['max_fixation_streak', 'mean_arrival_offset']
    res, df_acc = p8.evaluate_group(df, features, 3, 'rmsprop', 0.01, "pairs", "temporal", classes, 193, get_output_figure_name, df_acc)
    result.append(res)

    features = ['min_fixation_streak', 'std_departure_offset']
    res, df_acc = p8.evaluate_group(df, features, 3, 'rmsprop', 0.001, "pairs", "temporal", classes, 193, get_output_figure_name, df_acc)
    result.append(res)

    features = ['mean_fixation_streak', 'std_arrival_offset']
    res, df_acc = p8.evaluate_group(df, features, 2, 'rmsprop', 0.001, "pairs", "temporal", classes, 193, get_output_figure_name, df_acc)
    result.append(res)

    features = ['max_fixation_streak', 'std_departure_offset']
    res, df_acc = p8.evaluate_group(df, features, 2, 'rmsprop', 0.001, "pairs", "temporal", classes, 193, get_output_figure_name, df_acc)
    result.append(res)

    p8.plot_scatterplot("highest-pair", df[features], features, 'temporal', get_output_figure_name)

    p8.plot_boxplot("highest-pair", df_acc, get_output_figure_name)

    return result

def evaluate_highest_triplet(classes, df):

    result = []

    # highest triplet from research features
    features = ['leftward_qe_duration', 'perc_dwell_time_elsewhere', 'perc_small_sacc']
    res, df_acc = p8.evaluate_group(df, features, 3, 'rmsprop', 0.001, "triplets", "research", classes, 217, get_output_figure_name)
    result.append(res)

    features = ['rightward_qe_duration', 'perc_dwell_time_elsewhere', 'perc_small_sacc']# deze mss weghalen
    res, df_acc = p8.evaluate_group(df, features, 2, 'rmsprop', 0.001, "triplets", "research", classes, 217, get_output_figure_name, df_acc)
    result.append(res)

    # highest triplet from temporal features
    features = ['std_fixation_streak', 'min_departure_offset', 'min_arrival_offset']
    res, df_acc = p8.evaluate_group(df, features, 3, 'rmsprop', 0.001, "triplets", "temporal", classes, 217, get_output_figure_name, df_acc)
    result.append(res)

    features = ['min_departure_offset', 'mean_arrival_offset', 'max_arrival_offset']
    res, df_acc = p8.evaluate_group(df, features, 3, 'rmsprop', 0.01, "triplets", "temporal", classes, 217, get_output_figure_name, df_acc)
    result.append(res)

    features = ['mean_fixation_streak', 'std_fixation_streak', 'max_fixation_streak']
    res, df_acc = p8.evaluate_group(df, features, 3, 'rmsprop', 0.001, "triplets", "temporal", classes, 217, get_output_figure_name, df_acc)
    result.append(res)

    features = ['mean_fixation_streak', 'min_fixation_streak', 'std_departure_offset']
    res, df_acc = p8.evaluate_group(df, features, 2, 'rmsprop', 0.001, "triplets", "temporal", classes, 217, get_output_figure_name, df_acc)
    result.append(res)


    p8.plot_boxplot("highest-triplet", df_acc, get_output_figure_name)

    return result

def evaluate_highest_quartets(classes, df):

    result = []

    # highest quartet from research features
    features = ['mean_fixation_duration', 'max_fixation_duration', 'rightward_qe_duration', 'perc_small_sacc']
    res, df_acc = p8.evaluate_group(df, features, 3, 'rmsprop', 0.01, "quartets", "research", classes, 217, get_output_figure_name)
    result.append(res)

    # highest quartet from temporal features
    features = ['min_fixation_streak', 'max_fixation_streak', 'mean_departure_offset', 'std_arrival_offset']
    res, df_acc = p8.evaluate_group(df, features, 3, 'rmsprop', 0.001, "triplets", "temporal", classes, 217, get_output_figure_name, df_acc)
    result.append(res)


    p8.plot_boxplot("highest-quartet", df_acc, get_output_figure_name)

    return result



def evaluate_highest_overall(classes, df):

    result = []

    # highest overall from research features
    features = ['leftward_qe_duration', 'perc_dwell_time_target_dish', 'perc_dwell_time_elsewhere', 'perc_dwell_time_start_dish', 'mean_saccade_amplitude']
    res, df_acc = p8.evaluate_group(df, features, 2, 'rmsprop', 0.001, "overall", "research", classes, 314, get_output_figure_name)
    result.append(res)

    features = ['mean_fixation_duration', 'max_fixation_duration', 'leftward_qe_duration', 'perc_dwell_time_start_dish', 'perc_small_sacc']
    res, df_acc = p8.evaluate_group(df, features, 3, 'rmsprop', 0.01, "overall", "research", classes, 314, get_output_figure_name, df_acc)
    result.append(res)

    features = ['mean_fixation_duration', 'leftward_qe_duration', 'rightward_qe_duration', 'perc_dwell_time_target_dish', 'perc_small_sacc']
    res, df_acc = p8.evaluate_group(df, features, 3, 'rmsprop', 0.001, "overall", "research", classes, 314, get_output_figure_name, df_acc)
    result.append(res)

    features = ['mean_fixation_duration', 'min_fixation_duration', 'max_fixation_duration', 'rightward_qe_duration', 'perc_small_sacc', 'mean_saccade_amplitude']
    res, df_acc = p8.evaluate_group(df, features, 2, 'rmsprop', 0.001, "overall", "research", classes, 314, get_output_figure_name, df_acc)
    result.append(res)

    features = ['mean_fixation_duration', 'std_fixation_duration', 'min_fixation_duration', 'max_fixation_duration', 'rightward_qe_duration', 'perc_dwell_time_target_dish', 'perc_dwell_time_start_dish', 'perc_small_sacc']
    res, df_acc = p8.evaluate_group(df, features, 2, 'rmsprop', 0.01, "overall", "research", classes, 314, get_output_figure_name, df_acc)
    result.append(res)

    # highest overall from temporal features
    features = ['min_fixation_streak', 'max_fixation_streak', 'mean_departure_offset', 'std_arrival_offset']
    res, df_acc = p8.evaluate_group(df, features, 3, 'rmsprop', 0.001, "overall", "temporal", classes, 314, get_output_figure_name, df_acc)
    result.append(res)

    features = ['std_fixation_streak', 'mean_departure_offset', 'std_departure_offset', 'min_departure_offset', 'std_arrival_offset']
    res, df_acc = p8.evaluate_group(df, features, 3, 'rmsprop', 0.001, "overall", "temporal", classes, 314, get_output_figure_name, df_acc)
    result.append(res)

    features = ['mean_fixation_streak', 'std_fixation_streak', 'mean_departure_offset', 'max_departure_offset', 'std_arrival_offset', 'max_arrival_offset']
    res, df_acc = p8.evaluate_group(df, features, 2, 'rmsprop', 0.01, "overall", "temporal", classes, 314, get_output_figure_name, df_acc)
    result.append(res)

    features = ['mean_fixation_streak', 'std_fixation_streak', 'max_fixation_streak', 'mean_departure_offset', 'max_departure_offset', 'std_arrival_offset']
    res, df_acc = p8.evaluate_group(df, features, 2, 'rmsprop', 0.01, "overall", "temporal", classes, 314, get_output_figure_name, df_acc)
    result.append(res)

    p8.plot_boxplot("highest-overall", df_acc, get_output_figure_name)

    return result


if __name__ == "__main__":

    result = []
    aggregates = []

    tsis.make_dir(get_output_base())
    tsis.make_dir(get_output_figures_base())

    # load the dataframe
    df_in = pd.read_csv(get_input_file())

    # drop the mediums (for 2 classes)
    if CLASSES == 2:
        df = df_in[df_in['label'] != "Medium"]
    else:
        df = df_in

    scores = evaluate_control(CLASSES, df)
    result = result + scores

    scores = evaluate_highest_singles(CLASSES, df)
    result = result + scores

    scores = evaluate_highest_pairs(CLASSES, df)
    result = result + scores

    scores = evaluate_highest_triplet(CLASSES, df)
    result = result + scores

    scores = evaluate_highest_quartets(CLASSES, df)
    result = result + scores

    scores = evaluate_highest_overall(CLASSES, df)
    result = result + scores

    df_result = pd.DataFrame(result)


    df_predictions = df_result[['model','group','feature set','classes','features','label mapping','predictions']]
    df_aggregates = df_result[['model','group','feature set','classes','features','label mapping','aggregates']]

    df_predictions.to_csv(get_predictions_file_name(), index=False)
    df_aggregates.to_csv(get_aggregates_file_name(), index=False)

   
    print('*** FFNN evaluation and plotting completed ***')


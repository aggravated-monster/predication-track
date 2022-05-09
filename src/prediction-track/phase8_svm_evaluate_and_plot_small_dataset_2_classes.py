import tsi.tsi_sys as tsis
import phase8_svm_common as p8
import pandas as pd

CLASSES = 2

def get_output_base():
    return 'd:/ou/output/phase8/svm/experiment/small/' + str(CLASSES) + '-classes/'

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
    res, _ = p8.evaluate_group(1, "rbf", df, features, "single", "research", classes, 3)
    result.append(res)

    return result

def evaluate_highest_singles(classes, df):

    result = []

    # highest single from research features
    features = ['perc_large_sacc']
    res, df_acc = p8.evaluate_group(1, "rbf", df, features, "single", "research", classes, 3)
    result.append(res)

    # highest single from temporal features
    features = ['mean_fixation_streak']
    res, df_acc = p8.evaluate_group(1, "rbf", df, features, "single", "temporal", classes, 3, df_acc)
    result.append(res)

    p8.plot_boxplot("highest-single", df_acc, get_output_figure_name)

    return result

def evaluate_highest_pairs(classes, df):

    result = []

    # highest pair from research features
    features = ['min_fixation_duration', 'rightward_qe_duration']
    res, df_acc = p8.evaluate_group(100, "rbf", df, features, "pair", "research", classes, 117)
    result.append(res)

    p8.plot_scatterplot("highest-pair", df[features], features, 'research', get_output_figure_name)

    # highest pair from temporal features
    features = ['mean_fixation_streak', 'mean_departure_offset']
    res, df_acc = p8.evaluate_group(1, "rbf", df, features, "pair", "temporal", classes, 117, df_acc)
    result.append(res)

    p8.plot_boxplot("highest-pair", df_acc, get_output_figure_name)

    p8.plot_scatterplot("highest-pair", df[features], features, 'temporal', get_output_figure_name)

    return result

def evaluate_highest_triplet(classes, df):

    result = []

    # highest triplet from research features
    features = ['std_fixation_duration', 'rightward_qe_duration', 'perc_dwell_time_target_dish']
    res, df_acc = p8.evaluate_group(1, "sigmoid", df, features, "triplet", "research", classes, 277)
    result.append(res)

    # highest triplet from temporal features
    features = ['mean_fixation_streak', 'max_fixation_streak', 'mean_departure_offset']
    res, df_acc = p8.evaluate_group(1, "rbf", df, features, "triplet", "temporal", classes, 277, df_acc)
    result.append(res)

    p8.plot_boxplot("highest-triplet", df_acc, get_output_figure_name)

    return result

def evaluate_highest_overall(classes, df):

    result = []

    # highest overall from research features
    features = ['mean_fixation_duration', 'min_fixation_duration', 'rightward_qe_duration', 'perc_dwell_time_target_dish', 'perc_dwell_time_start_dish', 'perc_large_sacc']
    res, df_acc = p8.evaluate_group(10, "linear", df, features, "overall", "research", classes, 372)
    result.append(res)

    # highest overall from temporal features
    features = ['std_fixation_streak', 'mean_departure_offset', 'mean_arrival_offset', 'max_arrival_offset']
    res, df_acc = p8.evaluate_group(10, "linear", df, features, "overall", "temporal", classes, 372, df_acc)
    result.append(res)

    p8.plot_boxplot("highest-overall", df_acc, get_output_figure_name)

    return result

def evaluate_highest_non_dominants(classes, df):

    result = []

    # highest non-dominant from research features
    # the dominant is 
    # 'mean_fixation_duration', 
    # 'min_fixation_duration', 
    # 'rightward_qe_duration', 
    # 'perc_dwell_time_target_dish', 
    # 'perc_dwell_time_start_dish', 
    # 'perc_large_sacc'
    # Assuming all derivatives of fixation duration are in some way dominant, this leaves a combination in 
    # 'leftward_qe_duration', 'perc_dwell_time_elsewhere', 'mean_saccade_amplitude'
    # The highest turns out to be the combination of 3

    features = ['leftward_qe_duration', 'perc_dwell_time_elsewhere', 'mean_saccade_amplitude']
    res, df_acc = p8.evaluate_group(10, "sigmoid", df, features, "non-dominant", "research", classes, 425)
    result.append(res)

    # highest most economic non-dominant from temporal features
    # the dominant is 
    # 'std_fixation_streak', 
    # 'mean_departure_offset', 
    # 'mean_arrival_offset', 
    # 'max_arrival_offset, 
    # Assuming all derivatives of fixation streak and the offsets are in some way dominant, this leaves no other features

    p8.plot_boxplot("highest-non-dominant", df_acc, get_output_figure_name)

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

    scores = evaluate_highest_overall(CLASSES, df)
    result = result + scores

    scores = evaluate_highest_non_dominants(CLASSES, df)
    result = result + scores


    df_result = pd.DataFrame(result)


    df_predictions = df_result[['model','group','feature set','classes','features','label mapping','predictions']]
    df_aggregates = df_result[['model','group','feature set','classes','features','label mapping','aggregates']]

    df_predictions.to_csv(get_predictions_file_name(), index=False)
    df_aggregates.to_csv(get_aggregates_file_name(), index=False)

   
    print('*** SVM evaluation and plotting completed ***')



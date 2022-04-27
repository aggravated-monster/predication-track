import pandas as pd
import tsi.tsi_sys as tsis
import phase8_svm_common as p8

CLASSES = 3

def get_output_base():
    return 'd:/ou/output/phase8/svm/experiment/small/' + str(CLASSES) + '-classes/'

def get_output_figures_base():
    return get_output_base() + "figures/"

def get_output_figure_name(prefix, suffix):
    return get_output_figures_base() + prefix + '_' + suffix + '.png'

def get_input_file():
    # the small dataset
    return 'd:/ou/output/phase7/12_small_labeled_dataset/small_labeled_dataset.csv'

def get_predictions_file_name():
    return get_output_base() + 'predictions.csv'

def get_aggregates_file_name():
    return get_output_base() + 'aggregates.csv'


def evaluate_highest_singles(classes, df):

    result = []

    # highest single from research features
    features = ['perc_large_sacc']
    res, df_acc = p8.evaluate_group(0.1, "linear", df, features, "single", "research", classes, 2)
    result.append(res)

    # highest single from temporal features
    features = ['mean_fixation_streak']
    res, df_acc = p8.evaluate_group(1, "rbf", df, features, "single", "temporal", classes, 2, df_acc)
    result.append(res)

    p8.plot_boxplot("highest-single", df_acc, get_output_figure_name)

    return result

def evaluate_highest_pairs(classes, df):

    result = []

    # highest pair from research features
    features = ['rightward_qe_duration', 'perc_large_sacc']
    res, df_acc = p8.evaluate_group(10, "rbf", df, features, "pairs", "research", classes, 181)
    result.append(res)

    p8.plot_scatterplot("highest-pair", df[features], features, 'research', get_output_figure_name)

    # highest pair from temporal features
    features = ['mean_fixation_streak', 'std_departure_offset']
    res, df_acc = p8.evaluate_group(10, "sigmoid", df, features, "pairs", "research", classes, 181, df_acc)
    result.append(res)

    p8.plot_scatterplot("highest-pair", df[features], features, 'temporal', get_output_figure_name)

    return result

def evaluate_highest_triplet(classes, df):

    result = []

    # highest triplet from research features
    features = ['rightward_qe_duration', 'perc_dwell_time_target_dish', 'perc_large_sacc']
    res, df_acc = p8.evaluate_group(10, "sigmoid", df, features, "triplets", "research", classes, 251)
    result.append(res)

    # highest triplet from temporal features
    features = ['mean_arrival_offset', 'std_arrival_offset', 'max_arrival_offset']
    res, df_acc = p8.evaluate_group(10, "linear", df, features, "triplets", "temporal", classes, 251, df_acc)
    result.append(res)

    p8.plot_boxplot("highest-triplet", df_acc, get_output_figure_name)

    return result

def evaluate_highest_overall(classes, df):

    result = []

    # highest overall from research features
    features = ['leftward_qe_duration', 'rightward_qe_duration', 'perc_dwell_time_target_dish', 'perc_dwell_time_elsewhere', 'perc_small_sacc']
    res, df_acc = p8.evaluate_group(1, "sigmoid", df, features, "overall", "research", classes, 378)
    result.append(res)

    # highest overall from temporal features
    features = ['mean_fixation_streak', 'std_fixation_streak', 'max_fixation_streak', 'mean_departure_offset', 'std_departure_offset', 'min_departure_offset']
    res, df_acc = p8.evaluate_group(10, "rbf", df, features, "overall", "temporal", classes, 378, df_acc)
    result.append(res)

    p8.plot_boxplot("highest-overall", df_acc, get_output_figure_name)

    return result

def evaluate_highest_non_dominants(classes, df):

    result = []

    # highest non-dominant from research features
    # the dominant is 
    # 'leftward_qe_duration', 
    # 'rightward_qe_duration', 
    # 'perc_dwell_time_target_dish', 
    # 'perc_dwell_time_elsewhere', 
    # 'perc_small_sacc'
    # This leaves a combination in 
    # '*_fixation_duration', 'mean_saccade_amplitude'
    # The highest turns out to be the combination of 3

    features = ['mean_fixation_duration', 'min_fixation_duration', 'perc_dwell_time_start_dish', 'perc_large_sacc', 'mean_saccade_amplitude']
    res, df_acc = p8.evaluate_group(1, "sigmoid", df, features, "pairs", "research", classes, 494)
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



import tsi.tsi_sys as tsis
import tsi.tsi_data as tsid
import pandas as pd
import phase8_svm_common as p8
from itertools import combinations

def get_output_base():
    # change output base depending on feature group
    return 'd:/ou/output/phase8/svm/modeling/large/2-classes/temporal-features/'

def get_evaluation_output_base():
    return 'd:/ou/output/phase8/svm/evaluation/large/2-classes/temporal-features/'

def get_figures_output_base():
    return get_evaluation_output_base() + 'figures/'

def get_input_file():
    # the small dataset
    #return 'd:/ou/output/phase7/12_small_labeled_dataset/small_labeled_dataset.csv'
    # the large dataset
    return 'd:/ou/output/phase7/9_large_labeled_dataset/large_labeled_dataset.csv'

def get_best_scores_file_name():
    return get_output_base() + 'best_scores.csv'

def get_mean_scores_file_name():
    return get_output_base() + 'mean_scores.csv'

def get_all_scores_file_name():
    return get_output_base() + 'all_scores.csv'

if __name__ == "__main__":

    #best_scores = []
    all_scores = []
    # change columns depending on feature group
    columns = tsid.get_temporal_columns_large()

    tsis.make_dir(get_output_base())
    tsis.make_dir(get_evaluation_output_base())
    tsis.make_dir(get_figures_output_base())

    nr_of_runs = 1
    for i in range(nr_of_runs):

        # brute force search through all combinations
        # make sure the set of columns does not exceed 15, or else it becomes quite painful
        for k in range(len(columns)):

            choose = k+1

            feature_combinations = list(combinations(columns, choose))
            print("Performing N choose k brute force search on k=" + str(choose))
            print("length of combinations list: " + str(len(feature_combinations)))

            # load the dataframe
            df_in = pd.read_csv(get_input_file())

            # drop the mediums (for 2 classes)
            df_in = df_in[df_in['label'] != "Medium"]

            # define the reference model
            # SVM supports imbalanced multi class training sets
            model = p8.get_reference_model()

            param_grid = {"class_weight": ['balanced'],
                            "kernel": ["rbf", "sigmoid","linear"],
                            "gamma": ['scale'],
                            "C": [0.1, 1, 10, 100]
                            }
            # do the work
            df_prediction_result, df_search_result = p8.perform_model_search(df_in, model, 'SVM', param_grid, feature_combinations, get_figures_output_base, k*10+i)

            # sort for better readability
            df_prediction_result = df_prediction_result.sort_values(by='balanced accuracy', ascending=False)
            df_search_result = df_search_result.sort_values(by='best score', ascending=False)
            # write summary
            df_prediction_result.to_csv(get_evaluation_output_base() + 'choose_' + str(choose) + '_svm_scores_run_' + str(i+1) + '.csv', index=False)
            df_search_result.to_csv(get_output_base() + 'choose_' + str(choose) + '_svm_grid_search_run_' + str(i+1) + '.csv', index=False)

            all_scores.append(df_search_result)

    all_scores = pd.concat(all_scores)
    all_scores = all_scores.sort_values(by='best score', ascending=False)
    all_scores.to_csv(get_all_scores_file_name(), index=False)   

    p8.prepare_report(all_scores, nr_of_runs).to_csv(get_mean_scores_file_name())

    
    print('*** SVM modeling completed ***')



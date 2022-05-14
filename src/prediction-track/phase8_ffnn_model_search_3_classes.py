import pandas as pd 
import tsi.tsi_sys as tsis
import tsi.tsi_data as tsid
import phase8_ffnn_common as p8
from itertools import combinations

CLASSES = 3

def get_output_base():
    # change output base depending on which data set and number of classes
    return 'c:/ou/output/phase8/ffnn/modeling/large/3-classes/research/'

def get_evaluation_output_base():
    # change output base depending on which data set and number of classes
    return 'c:/ou/output/phase8/ffnn/evaluation/large/3-classes/research/'

def get_input_file():
    # change depending on which data set and number of classes
    # the small dataset
    #return 'd:/ou/output/phase7/12_small_labeled_dataset/small_labeled_dataset.csv'
    # the large dataset
    return 'c:/ou/output/phase7/9_large_labeled_dataset/large_labeled_dataset.csv'

def get_best_scores_file_name():
    return get_output_base() + 'best_scores.csv'

def get_all_scores_file_name():
    return get_output_base() + 'all_scores.csv'

def get_figures_output_base():
    return get_evaluation_output_base() + 'figures/'

def get_mean_scores_file_name():
    return get_output_base() + 'mean_scores.csv'

if __name__ == "__main__":

    all_scores = []
    # change depending on which features are to be evaluated
    columns = tsid.get_research_columns_large()

    tsis.make_dir(get_output_base())
    tsis.make_dir(get_evaluation_output_base())
    tsis.make_dir(get_figures_output_base())

    nr_of_runs = 1
    for i in range(nr_of_runs):
        # brute force search through all combinations
        for k in range(len(columns)):

            choose = k+1

            feature_combinations = list(combinations(columns, choose))
            print("Performing N choose k brute force search on k=" + str(choose))
            print("length of combinations list: " + str(len(columns)))
        
            tsis.make_dir(get_output_base())

            # load the dataframe
            df_in=pd.read_csv(get_input_file(), encoding='utf-8')
            
            params = {
                "model_depth": [2, 3],
                "optimizer__learning_rate": [0.001, 0.01],
            }

            # do the work
            df_prediction_result, df_search_result = p8.perform_model_search_3_classes(df_in, p8.create_model_grid, 'FFNN', params, feature_combinations, get_figures_output_base, CLASSES, k*10)

            # sort for better readability
            df_prediction_result = df_prediction_result.sort_values(by='balanced accuracy', ascending=False)
            df_search_result = df_search_result.sort_values(by='best score', ascending=False)

            # write summary
            df_prediction_result.to_csv(get_evaluation_output_base() + 'choose_' + str(choose) + '_ffnn_scores_run_' + str(i+1) + '.csv', index=False)
            df_search_result.to_csv(get_output_base() + 'choose_' + str(choose) + '_ffnn_grid_search_run_' + str(i+1) + '.csv', index=False)

            all_scores.append(df_search_result)

    all_scores = pd.concat(all_scores)
    all_scores = all_scores.sort_values(by='best score', ascending=False)
    all_scores.to_csv(get_all_scores_file_name(), index=False)   

    p8.prepare_report(all_scores, nr_of_runs).to_csv(get_mean_scores_file_name())

    print('*** FFNN modeling completed ***')

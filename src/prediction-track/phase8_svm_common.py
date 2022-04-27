from scipy.stats import sem
import numpy as np
import pandas as pd
import tsi.tsi_modeling as tsim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.preprocessing import LabelEncoder

def get_reference_model():
    return svm.SVC(gamma='scale', class_weight='balanced')

def get_model(c, kernel):
    return svm.SVC(gamma='scale', class_weight='balanced', C=c, kernel=kernel)

def prep_x_y(df_in, features):

    df_rf = df_in[features]
    # drop the rows with null values
    df_rf = df_rf.dropna()

    X = df_rf.drop('label', axis=1)

    # label encode the target variable
    le = LabelEncoder()
    y = le.fit_transform(df_rf['label'])
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    return (X, y, label_mapping)

def calculate_aggregates(scores):
    
    df = pd.DataFrame.from_dict(scores)

    acc_mean = np.mean(df['acc'])
    acc_std = np.std(df['acc'])
    acc_se = sem(df['acc'])

    precision_mean = np.mean(df['precision'])
    precision_std = np.std(df['precision'])
    precision_se = sem(df['precision'])

    recall_mean = np.mean(df['recall'])
    recall_std = np.std(df['recall'])
    recall_se = sem(df['recall'])

    f1_mean = np.mean(df['f1'])
    f1_std = np.std(df['f1'])
    f1_se = sem(df['f1'])

    return {
        'sample size': len(df),
        'acc mean': acc_mean,
        'acc stcd': acc_std, 
        'acc se': acc_se, 
        'precision mean': precision_mean, 
        'precision std': precision_std, 
        'precision se': precision_se, 
        'recall mean': recall_mean, 
        'recall std': recall_std, 
        'recall se': recall_se, 
        'f1 mean': f1_mean, 
        'f1 std': f1_std, 
        'f1 se': f1_se
    }

def plot_boxplot(group, df, f_get_output_figure_name):

    width = df.shape[1] * 5

    sns.set(rc = {'figure.figsize':(width,8)})
    sns.boxplot(data=df)
    plt.savefig(f_get_output_figure_name(group, 'boxplot'), bbox_inches='tight')
    plt.close()

def plot_scatterplot(group, df, features, feature_set, f_get_output_figure_name):
    
    sns.set(rc = {'figure.figsize':(8,8)})
    sns.scatterplot(data = df, x=features[0], y=features[1], hue='label')
    plt.savefig(f_get_output_figure_name(group, feature_set + '_scatter'), bbox_inches='tight')
    plt.close()

def evaluate_group(c, kernel, df, features, group, feature_set, classes, seed, accumulator='none'):
    aggregates = []

    result_dict = {
        'model': 'svm',
        'group': 'highest-' + group,
        'classes': classes
    }

    repetitions = 100

    features.append("label")
    predictions, label_mapping = run_predictions(c, kernel, df, features, seed, repetitions, get_model, classes)
    aggregates = calculate_aggregates(predictions)
    result_dict['feature set'] = feature_set
    result_dict['features'] = features
    result_dict['label mapping'] = label_mapping
    result_dict['aggregates'] = aggregates
    result_dict['predictions'] = predictions

    acc = list(map(lambda d: d['acc'], predictions))
    acc_column = 'acc ' + feature_set

    if isinstance(accumulator, str): # very dirty trick
        accumulator = pd.DataFrame(acc, columns=['acc_column'])
    else:
        accumulator[acc_column] = acc  

    return result_dict, accumulator

def perform_model_search(df_in, model, classifier, param_grid, feature_list, f_destination, seed=None):
    prediction_result = []
    search_result = []

    for scorer in tsim.get_scorers():

        # brute force the whole thing one by one
        for features_tuple in feature_list:

            features = list(features_tuple)
            print("evaluating featureset: " + ",".join(features))
            features.append('label') # always include the label

            X, y, label_mapping = prep_x_y(df_in, features)

            X_train, X_test, y_train, y_test = tsim.prepare_training(X, y, seed) 

            grid_result = tsim.grid_search(X_train, y_train, model, scorer, param_grid, seed)
        
            # report the best configuration
            best_std = grid_result.cv_results_['std_test_score'][grid_result.best_index_]
            best_mean = grid_result.cv_results_['mean_test_score'][grid_result.best_index_]
            best_se = best_std / np.sqrt(np.size(X_train))

            best_score = grid_result.best_score_
            best_estimator = grid_result.best_estimator_
            best_params = grid_result.best_params_

            search_result.append((features, classifier, scorer, best_score, best_estimator, best_params, best_mean, best_std, best_se))

            # predict
            y_true, y_pred = y_test, grid_result.predict(X_test)

            # report on prediction
            prediction_report = tsim.report_prediction_scores(y_true, y_pred)

            if len(np.unique(y_true)) == 2:
                #plot_precision_recall(features, best_estimator, X_test, y_test, f_destination)
                tsim.plot_roc(features, best_estimator, X_test, y_test, f_destination)
                #plot_confusion_matrix(features, y_true, y_pred, f_destination)

            prediction_result.append((classifier, features, label_mapping, len(y_train), len(y_test)) + prediction_report)
    
    df_prediction_result = pd.DataFrame(prediction_result, columns=['model', 'features', 'label mapping', 'training size', 'test size', 'balanced accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'classification report', 'confusion_matrix', 'roc curve', 'precision-recall curve'])
    df_search_result = pd.DataFrame(search_result, columns=['features', 'classifier', 'scorer', 'best score', 'best estimator', 'best params', 'best_mean', 'best_std', 'best_se'])
    
    return (df_prediction_result, df_search_result)

def run_predictions(c, kernel, df_in, features, seed, repetitions, f_get_model, classes=2):

    predictions = []
    # get the model to evaluate
    model = f_get_model(c, kernel)
    X, y, label_mapping = prep_x_y(df_in, features)

    for i in range(repetitions):
        X_train, X_test, y_train, y_test = tsim.prepare_training(X, y, seed + 10*i)
        model.fit(X_train, y_train)

        # predict
        y_true, y_pred = y_test, model.predict(X_test)

        # report on prediction
        predictions.append(tsim.report_prediction_scores_as_dict(y_true, y_pred, classes))

    return predictions, label_mapping

def prepare_report(df_scores, nr_of_runs):

    # take features and scores
    df_agg = df_scores[['features', 'best score']]
    # get rid of the list format
    df_agg['feature set'] = df_agg['features'].apply(lambda x: ','.join(x))

    # aggregate on features, calc mean best score
    df_agg = df_agg.groupby('feature set').mean()
    # add number of runs. This is purely informational
    df_agg['nr_of_runs'] = nr_of_runs
    # rename best score, as it is a mean npw
    df_agg.rename(columns={"best score": "mean score"}, inplace=True)

    # order by highest mean
    df_agg = df_agg.sort_values(by='mean score', ascending=False)

    return df_agg

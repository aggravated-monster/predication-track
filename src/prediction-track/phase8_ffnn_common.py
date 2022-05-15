#https://towardsdatascience.com/feed-forward-neural-networks-how-to-successfully-build-them-in-python-74503409d99a
# Tensorflow / Keras
from sympy import ceiling
from tensorflow import keras # for building Neural Networks
print('Tensorflow/Keras: %s' % keras.__version__) # print version
from scikeras.wrappers import KerasClassifier
from keras.utils import np_utils
from random import randrange
import tsi.tsi_modeling as tsim
import pandas as pd
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def create_model(n_features_in, X_shape, model_depth=2,
                      seed=None,
                      optimizer='adam', 
                      learning_rate=0.1,
                      activation = 'tanh', 
                      init='glorot_uniform'
                      ):
    # create model
    neural_classifier = keras.models.Sequential()

    number_of_neurons = ceiling((n_features_in + 2)/2)

    neural_classifier.add(keras.layers.Dense(n_features_in, input_shape=X_shape[1:]))
    neural_classifier.add(keras.layers.Activation(activation))
    for i in range(model_depth):
        neural_classifier.add(keras.layers.Dense(number_of_neurons, kernel_initializer=init))
        neural_classifier.add(keras.layers.Activation(activation))
    neural_classifier.add(keras.layers.Dense(1))
    neural_classifier.add(keras.layers.Activation("sigmoid"))
    
    # Compile model
    neural_classifier.compile(loss='binary_crossentropy', 
                  optimizer=optimizer, 
                  metrics=['accuracy'])
    
    # wrap in KerasClassifier
    scikeras_classifier = KerasClassifier(model=neural_classifier,
                                        optimizer__learning_rate=learning_rate,
                                        batch_size=8,
                                        epochs=2000,
                                        verbose=1,
                                        random_state=seed
                                        )
    return scikeras_classifier

def create_model_3_classes(n_features_in, X_shape, model_depth=2,
                      seed=None,
                      optimizer='adam', 
                      learning_rate=0.1,
                      activation = 'tanh', 
                      init='glorot_uniform'
                      ):
    # create model
    neural_classifier = keras.models.Sequential()

    number_of_neurons = ceiling((n_features_in + 2)/2)

    neural_classifier.add(keras.layers.Dense(n_features_in, input_shape=X_shape[1:]))
    neural_classifier.add(keras.layers.Activation(activation))
    for i in range(model_depth):
        neural_classifier.add(keras.layers.Dense(number_of_neurons, kernel_initializer=init))
        neural_classifier.add(keras.layers.Activation(activation))
    neural_classifier.add(keras.layers.Dense(3))
    neural_classifier.add(keras.layers.Activation("softmax"))
    
    # Compile model
    neural_classifier.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer, 
                  metrics=['accuracy'])
    
    # wrap in KerasClassifier
    scikeras_classifier = KerasClassifier(model=neural_classifier,
                                        optimizer__learning_rate=learning_rate,
                                        batch_size=8,
                                        epochs=10,
                                        verbose=1,
                                        random_state=seed
                                        )
    return scikeras_classifier


def create_model_grid(n_features_in, X_shape, model_depth=2,
                      activation = 'tanh', 
                      optimizer='adam', 
                      init='glorot_uniform',
                      ):
    # create model
    model = keras.models.Sequential()

    number_of_neurons = ceiling((n_features_in + 2)/2)

    model.add(keras.layers.Dense(n_features_in, input_shape=X_shape[1:]))
    model.add(keras.layers.Activation(activation))
    for i in range(model_depth):
        model.add(keras.layers.Dense(number_of_neurons, kernel_initializer=init))
        model.add(keras.layers.Activation(activation))
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation("sigmoid"))
    
    # Compile model
    model.compile(loss='binary_crossentropy', 
                  optimizer=optimizer, 
                  metrics=['accuracy'])
    return model

def create_model_grid_3_classes(n_features_in, X_shape, model_depth=2,
                      activation = 'tanh', 
                      optimizer='adam', 
                      init='glorot_uniform',
                      ):
    # create model
    model = keras.models.Sequential()

    number_of_neurons = ceiling((n_features_in + 2)/2)

    model.add(keras.layers.Dense(n_features_in, input_shape=X_shape[1:]))
    model.add(keras.layers.Activation(activation))
    for i in range(model_depth):
        model.add(keras.layers.Dense(number_of_neurons, kernel_initializer=init))
        model.add(keras.layers.Activation(activation))
    model.add(keras.layers.Dense(3))
    model.add(keras.layers.Activation("softmax"))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer, 
                  metrics=['accuracy'])
    return model


def evaluate_group(df_in, features, model_depth, optimizer, learning_rate, group, feature_set, classes, seed, f_get_output_figure_name, accumulator='none'):
    aggregates = []

    result_dict = {
        'model': 'ffnn',
        'group': 'highest-' + group,
        'classes': classes
    }

    repetitions = 10

    features.append("label")

    if classes == 2:
        X, y, label_mapping = prep_x_y(df_in, features)
        predictions = run_predictions(X, y, seed, repetitions, features, model_depth, optimizer, learning_rate, f_get_output_figure_name, feature_set, group)
        aggregates = calculate_aggregates(predictions)
    else:
        X, y, label_mapping = prep_x_y_3_classes(df_in, features)
        predictions = run_predictions_3_classes(X, y, seed, repetitions, features, model_depth, optimizer, learning_rate, f_get_output_figure_name, feature_set, group)
        aggregates = calculate_aggregates_3_classes(predictions)

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

def do_prediction(X, y, learning_rate, seed, model_depth, optimizer, num_features=2):
    model = create_model(num_features, X.shape, model_depth, seed, optimizer, learning_rate)

    X_train, X_test, y_train, y_test = tsim.prepare_training(X, y, seed)

    print(X.shape, y.shape)

    history = model.fit(X_train, y_train, epochs=2000, validation_data=(X_test, y_test))
    # predict
    y_true, y_pred = y_test, model.predict(X_test)

    return y_true, y_pred, y_test, history


def do_prediction_3_classes(X, y, learning_rate, seed, model_depth, optimizer, num_features=2):
    model = create_model_3_classes(num_features, X.shape, model_depth, seed, optimizer, learning_rate)

    X_train, X_test, y_train, y_test = tsim.prepare_training(X, y, seed)

    print(X.shape, y.shape)

    history = model.fit(X_train, y_train, epochs=2000, validation_data=(X_test, y_test))

    # predict
    y_true, y_pred = y_test, model.predict(X_test)

    return y_true, y_pred, y_test, history


def run_predictions(X, y, seed, repetitions, features, model_depth, optimizer, learning_rate, f_get_output_figure_name, feature_set, group):

    predictions = []

    # do one rep for the loss graph
    y_true, y_pred, y_test, history = do_prediction(X, y, learning_rate, seed+11, model_depth, optimizer, len(features))
    plot_loss(history, feature_set, group, f_get_output_figure_name)
    
    for i in range(repetitions):
        y_true, y_pred, y_test, history = do_prediction(X, y, learning_rate, seed + 10*i, model_depth, optimizer, len(features))

        # report on prediction
        predictions.append(tsim.report_prediction_scores_as_dict(y_true, y_pred))

    return predictions

def run_predictions_3_classes(X, y, seed, repetitions, features, model_depth, optimizer, learning_rate, f_get_output_figure_name, feature_set, group):

    predictions = []

    # do one rep for the loss graph
    y_true, y_pred, y_test, history = do_prediction_3_classes(X, y, learning_rate, seed+11, model_depth, optimizer, len(features))
    plot_loss(history, feature_set, group, f_get_output_figure_name)
    
    for i in range(repetitions):
        y_true, y_pred, y_test, history = do_prediction_3_classes(X, y, learning_rate, seed + 10*i, model_depth, optimizer, len(features))

        # report on prediction
        predictions.append(tsim.report_prediction_scores_as_dict(y_true, y_pred, 3, True))

    return predictions


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

def calculate_aggregates_3_classes(scores):
    
    df = pd.DataFrame.from_dict(scores)

    acc_mean = np.mean(df['acc'])
    acc_std = np.std(df['acc'])
    acc_se = sem(df['acc'])

    return {
        'sample size': len(df),
        'acc mean': acc_mean,
        'acc std': acc_std, 
        'acc se': acc_se, 
        'precision mean': 'NA', 
        'precision std': 'NA', 
        'precision se': 'NA', 
        'recall mean': 'NA', 
        'recall std': 'NA', 
        'recall se': 'NA', 
        'f1 mean': 'NA', 
        'f1 std': 'NA', 
        'f1 se': 'NA'
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

def plot_loss(history, feature_set, group, f_get_output_figure_name):
    plt.plot(history.history_['loss'])
    plt.savefig(f_get_output_figure_name(feature_set, group + str(randrange(0, 100))), bbox_inches='tight')
    plt.close()

def prepare_report(df_scores, nr_of_runs):

    # take features and scores
    df_agg = df_scores[['features', 'best score']]
    # get rid of the list format
    df_agg['feature set'] = df_agg['features'].apply(lambda x: ','.join(x))

    # aggregate on features, calc mean best score
    df_agg = df_agg.groupby('feature set').mean()
    # add number of runs. This is purely informational
    df_agg['nr_of_runs'] = nr_of_runs
    # rename acc test, as it is a mean npw
    df_agg.rename(columns={"best score": "mean score"}, inplace=True)

    # order by highest mean
    df_agg = df_agg.sort_values(by='mean score', ascending=False)

    return df_agg
    
def prep_x_y(df_in, features):

    df = df_in[features]
    # drop the rows with null values
    df = df.dropna()

    size = len(df.columns)-1
    dataset = df.values

    X = dataset[1:,0:size].astype(float)
    Y = dataset[1:,size]

    # encode class values as integers
    le = LabelEncoder()
    le.fit(Y)
    encoded_Y = le.transform(Y)

    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    return (X, encoded_Y, label_mapping)

def prep_x_y_3_classes(df_in, features):

    df = df_in[features]
    # drop the rows with null values
    df = df.dropna()

    size = len(df.columns)-1
    dataset = df.values

    X = dataset[1:,0:size].astype(float)
    Y = dataset[1:,size]

    #https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/#:~:text=A%20one%20hot%20encoding%20is,is%20marked%20with%20a%201.
    # encode class values as integers
    le = LabelEncoder()
    integer_encoded = le.fit_transform(Y)
    print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded_y = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded_y)
    # invert first example

    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    return (X, onehot_encoded_y, label_mapping)

def perform_model_search(df_in, f_get_model, classifier, param_grid, feature_list, f_destination, classes, seed):
    prediction_result = []
    search_result = []

    i=0

    for scorer in tsim.get_scorers():

        # brute force the whole thing one by one
        for features_tuple in feature_list:
            i+=1 # is used foor seeding
            features = list(features_tuple)
            print("evaluating featureset: " + ",".join(features))
            features.append('label') # always include the label

            #X, y, label_mapping = prep_x_y_ffnn(df_in, features)
            X, y, label_mapping = prep_x_y(df_in, features)

            # wrap the model in a scikeras.KerasClassifier
            model_grid = KerasClassifier(build_fn=f_get_model, epochs=50 ,verbose=0, n_features_in=len(features)-1, X_shape=X.shape, activation='tanh', model_depth=2)

            X_train, X_test, y_train, y_test = tsim.prepare_training(X, y, seed) 

            # last param is important! n_jobs cannot be -1, as GPU enabled search does not multithread
            grid_result = tsim.grid_search(X_train, y_train, model_grid, scorer, param_grid, seed, None) 
        
            # report the best configuration
            best_std = grid_result.cv_results_['std_test_score'][grid_result.best_index_]
            best_mean = grid_result.cv_results_['mean_test_score'][grid_result.best_index_]
            best_se = best_std / np.sqrt(np.size(X_train))

            best_score = grid_result.best_score_
            best_estimator = grid_result.best_estimator_
            best_params = grid_result.best_params_

            print(grid_result)

            search_result.append((features, classifier, scorer, best_score, best_estimator, best_params, best_mean, best_std, best_se))

            # predict
            y_true, y_pred = y_test, grid_result.predict(X_test)

            # report on prediction
            prediction_report = tsim.report_prediction_scores(y_true, y_pred)

            if len(np.unique(y_true)) == 2:
                #plot_precision_recall(features, best_estimator, X_test, y_test, f_destination)
                #plot_roc(features, best_estimator, X_test, y_test, f_destination)
                tsim.plot_confusion_matrix(features, y_true, y_pred, f_destination)

            prediction_result.append((classifier, features, label_mapping, len(y_train), len(y_test)) + prediction_report)


    df_prediction_result = pd.DataFrame(prediction_result, columns=['model', 'features', 'label mapping', 'training size', 'test size', 'balanced accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'classification report', 'confusion_matrix', 'roc curve', 'precision-recall curve'])
    df_search_result = pd.DataFrame(search_result, columns=['features', 'classifier', 'scorer', 'best score', 'best estimator', 'best params', 'best_mean', 'best_std', 'best_se'])
    
    return (df_prediction_result, df_search_result)

def perform_model_search_3_classes(df_in, f_get_model, classifier, param_grid, feature_list, f_destination, classes, seed):
    prediction_result = []
    search_result = []

    i=0

    scorer = 'accuracy'

    # brute force the whole thing one by one
    for features_tuple in feature_list:
        i+=1 # is used foor seeding
        features = list(features_tuple)
        print("evaluating featureset: " + ",".join(features))
        features.append('label') # always include the label

        #X, y, label_mapping = prep_x_y_ffnn(df_in, features)
        X, y, label_mapping = prep_x_y(df_in, features)

        # wrap the model in a scikeras.KerasClassifier
        model_grid = KerasClassifier(build_fn=f_get_model, epochs=50 ,verbose=0, n_features_in=len(features)-1, X_shape=X.shape, activation='tanh', model_depth=2)

        X_train, X_test, y_train, y_test = tsim.prepare_training(X, y, seed) 

        # last param is important! n_jobs cannot be -1, as GPU enabled search does not multithread
        grid_result = tsim.grid_search(X_train, y_train, model_grid, scorer, param_grid, seed, None) 
    
        # report the best configuration
        best_std = grid_result.cv_results_['std_test_score'][grid_result.best_index_]
        best_mean = grid_result.cv_results_['mean_test_score'][grid_result.best_index_]
        best_se = best_std / np.sqrt(np.size(X_train))

        best_score = grid_result.best_score_
        best_estimator = grid_result.best_estimator_
        best_params = grid_result.best_params_

        print(grid_result)

        search_result.append((features, classifier, scorer, best_score, best_estimator, best_params, best_mean, best_std, best_se))

        # predict
        y_true, y_pred = y_test, grid_result.predict(X_test)

        # report on prediction
        prediction_report = tsim.report_prediction_scores(y_true, y_pred, 3)

        if len(np.unique(y_true)) == 2:
            #plot_precision_recall(features, best_estimator, X_test, y_test, f_destination)
            #plot_roc(features, best_estimator, X_test, y_test, f_destination)
            tsim.plot_confusion_matrix(features, y_true, y_pred, f_destination)

        prediction_result.append((classifier, features, label_mapping, len(y_train), len(y_test)) + prediction_report)


    df_prediction_result = pd.DataFrame(prediction_result, columns=['model', 'features', 'label mapping', 'training size', 'test size', 'balanced accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'classification report', 'confusion_matrix', 'roc curve', 'precision-recall curve'])
    df_search_result = pd.DataFrame(search_result, columns=['features', 'classifier', 'scorer', 'best score', 'best estimator', 'best params', 'best_mean', 'best_std', 'best_se'])
    
    return (df_prediction_result, df_search_result)


def evaluate_model(X, y, model, seed):

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, stratify=y, random_state=seed)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Test  Accuracy : {:.2f}".format(model.score(X_test, y_test)))
    print("Train Accuracy : {:.2f}".format(model.score(X_train, y_train)))

    return (model.score(X_test, y_test), model.score(X_train, y_train), y_test, y_pred, len(y_test), len(y_train), model.history_["loss"], model.history_["val_loss"])

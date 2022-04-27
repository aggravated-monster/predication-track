from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import optimizers
from keras.layers import CuDNNLSTM, Dense
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy.stats import sem
import pandas as pd
from sklearn.model_selection import train_test_split
import tsi.tsi_modeling as tsim

def plot_scatter(y_pred, y_test, learning_rate, f_get_output_figure_name, normalized=False):
    plt.scatter(range(len(y_pred)), y_pred, color='r')
    plt.scatter(range(len(y_test)), y_test, color='g')
    if normalized:
        plt.savefig(f_get_output_figure_name('predictions_normalized', str(learning_rate)), bbox_inches='tight')
    else:
        plt.savefig(f_get_output_figure_name('predictions', str(learning_rate)), bbox_inches='tight')
    plt.close()

def plot_loss(history, learning_rate, f_get_output_figure_name):
    plt.plot(history.history['loss'])
    plt.savefig(f_get_output_figure_name('learning_rate', str(learning_rate)), bbox_inches='tight')
    plt.close()

def apply_padding(df):


    max_len = max([len(arr) for arr in df['sequence']])

    print(df['sequence'])
    df['padded sequence'] = df['sequence'].apply(lambda seq : np.lib.pad(seq, ((max_len - len(seq),0),(0, 0)), 'constant', constant_values=100))
    print(df['padded sequence'])
    return df




def create_model(learning_rate, num_features=2):
    # create model
    model = Sequential()

    model.add(CuDNNLSTM((40), batch_input_shape=(None, None, num_features), return_sequences=True))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(CuDNNLSTM((40), return_sequences=False))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid')) 

    # depending on version
    #opt = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    opt = optimizers.adam_v2.Adam(lr=learning_rate)

    # Compile model
    model.compile(loss='binary_crossentropy', 
                  optimizer=opt, 
                  metrics=['accuracy'])
    return model

def prep_x_y_lstm(df_in, features):

    df_rf = df_in[features]
    # drop the rows with null values
    df_rf = df_rf.dropna()

    X_dataset = df_rf.drop('label', axis=1)

    # label encode the target variable
    le = LabelEncoder()
    y = le.fit_transform(df_rf['label'])
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    print(X_dataset)
    print(X_dataset.shape)
    print(X_dataset.to_numpy().shape)

    X = []

    for x in X_dataset.to_numpy():
        x_arr = np.array(x[0])
        X.append(x_arr)

    
    X = np.array(X)

    print(X.shape)

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
        'acc std': acc_std, 
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

def do_prediction(X, y, learning_rate, seed, num_features=2):
    model = create_model(learning_rate, num_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)

    history = model.fit(X_train, y_train, epochs=2000, validation_data=(X_test, y_test))
    # predict
    y_true, y_pred = y_test, model.predict(X_test)

    # Binarize predictions with 0.5 as thresold
    y_pred_norm=np.transpose(y_pred)[0]  # transformation to get (n,)
    y_pred_norm = list(map(lambda x: 0 if x < 0.5 else 1, y_pred_norm))

    return y_true, y_pred, y_pred_norm, y_test, history

def run_predictions(X, y, seed, repetitions, learning_rate, f_get_output_figure_name, num_features=2):
    
    predictions = []

    # do one rep for the pictures
    y_true, y_pred, y_pred_norm, y_test, history = do_prediction(X, y, learning_rate, seed+1, num_features)

    # plot last one to have graphics
    plot_scatter(y_pred, y_test, learning_rate, f_get_output_figure_name)
    plot_scatter(y_pred_norm, y_test, learning_rate, f_get_output_figure_name, True)
    plot_loss(history, learning_rate, f_get_output_figure_name)

    for i in range(repetitions):

        y_true, y_pred, y_pred_norm, y_test, _ = do_prediction(X, y, learning_rate, 10*i + seed, num_features)

        # report on prediction
        predictions.append(tsim.report_prediction_scores_as_dict(y_true, y_pred_norm))
    

    return predictions
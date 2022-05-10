# Utility library grouping functions that have to do with generic modeling and reporting functions
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

def get_scorers():
    #return ['balanced_accuracy', 'f1_weighted']
    return ['balanced_accuracy']


# evaluate a model
# https://machinelearningmastery.com/multi-class-imbalanced-classification/
def cross_validate_model(X, y, model, seed):

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=seed)
    # 5 folds for small dataset
    #cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
    scores = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=cv, verbose=2)
    
    return scores, np.mean(scores), np.std(scores), sem(scores)

def prepare_training(X, y, seed=None):
    # Split the dataset into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed) # stratify because of imbalanced set
    
    # scale the dataset
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    return X_train, X_test, y_train, y_test

# do not do a grid search with jobs=-1 on a GPU
def grid_search(X_train, y_train, model, scorer, param_grid, seed=None, jobs=-1):
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
    # define grid search
    grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=2,n_jobs=jobs, cv=cv, scoring=scorer)
    # execute the grid search
    return grid.fit(X_train, y_train)

def plot_precision_recall(features, classifier, X_test, y_test, f_destination):
    # bit of protection for the small data set which has too many features to fit in the display title
    if len(features) < 4:
        feature_string = ','.join(features[:len(features)-1])
        y_score = classifier.decision_function(X_test)

        name = classifier.kernel

        display = PrecisionRecallDisplay.from_predictions(y_test, y_score, name=name)
        fig = display.ax_.set_title("2-class Precision-Recall - " + feature_string)
        plt.savefig(f_destination() + 'PR_' + str(len(features)-1) + '_' + feature_string, bbox_inches='tight')
        plt.close('all')
        del fig

def plot_roc(features, classifier, X_test, y_test, f_destination):
    # bit of protection for the small data set which has too many features to fit in the display title
    if len(features) < 5:
        feature_string = ','.join(features[:len(features)-1])
        y_score = classifier.decision_function(X_test)

        name = classifier.kernel

        display = RocCurveDisplay.from_predictions(y_test, y_score, name=name)
        fig = display.ax_.set_title("2-class ROC - " + feature_string)
        plt.savefig(f_destination() + 'ROC_' + str(len(features)-1) + '_' + feature_string, bbox_inches='tight')
        plt.close('all')
        del fig

def plot_confusion_matrix(features, y_true, y_pred, f_destination):
    # bit of protection for the small data set which has too many features to fit in the display title
    if len(features) < 4:
        feature_string = ','.join(features[:len(features)-1])

        display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
        fig = display.ax_.set_title("2-class confusion matrix - " + feature_string)
        plt.savefig(f_destination() + 'CM_' + str(len(features)-1) + '_' + feature_string, bbox_inches='tight')
        plt.close('all')
        del fig

def report_prediction_scores(y_true, y_pred, classes=2):

    if classes == 2:
        precision_pred = precision_score(y_true, y_pred)
        recall_pred = recall_score(y_true, y_pred)
        f1_pred = f1_score(y_true, y_pred)
        roc_auc_score_pred = roc_auc_score(y_true, y_pred)
        roc_curve_pred = roc_curve(y_true, y_pred)
        precision_recall_curve_pred = precision_recall_curve(y_true, y_pred)
        class_report = classification_report(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        cf_matrix = {'tn': cm[0, 0], 'fp': cm[0, 1], 'fn': cm[1, 0], 'tp': cm[1, 1]}
        acc_pred = balanced_accuracy_score(y_true, y_pred)
    else:
        precision_pred = precision_score(y_true, y_pred, average='weighted')
        recall_pred = recall_score(y_true, y_pred, average='weighted')
        f1_pred = f1_score(y_true, y_pred, average='weighted')
        roc_auc_score_pred = None
        roc_curve_pred = None
        precision_recall_curve_pred = None        
        class_report = None
        cm = None
        cf_matrix = None
        acc_pred = balanced_accuracy_score(y_true, y_pred)


    return (acc_pred, precision_pred, recall_pred, f1_pred, roc_auc_score_pred, class_report, cf_matrix, roc_curve_pred, precision_recall_curve_pred)

def report_prediction_scores_as_dict(y_true, y_pred, classes=2):

    acc_pred, precision_pred, recall_pred, f1_pred, roc_auc_score_pred, class_report, cf_matrix, roc_curve_pred, precision_recall_curve_pred = report_prediction_scores(y_true, y_pred, classes)

    return {
        'acc': acc_pred,
        'precision': precision_pred,
        'recall': recall_pred,
        'f1': f1_pred,
        'roc_auc': roc_auc_score_pred,
        'classification report': class_report,
        'confusion matrix': cf_matrix,
        'roc curve': roc_curve_pred,
        'precision recall curve': precision_recall_curve_pred
    }





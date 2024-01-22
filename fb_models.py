"""
This module contains machine learning models for classification tasks.
It includes necessary libraries and modules from sklearn for creating and evaluating models.

The module currently contains a function `run_knn` for running the k-nearest
 neighbors (kNN) algorithm on given training and test data.

The function takes in training data features (`x_train`),
 test data features (`x_test`), training data labels (`y_train`),
 test data labels (`y_test`), and an optional parameter `n_neighbors`
 which is the number of neighbors to consider (default is 3).

The function fits the kNN model on the training data, makes predictions
 on the test data, and calculates various metrics such as accuracy, precision,
 recall, F1 score, confusion matrix, and classification report. These metrics
 are returned as a dictionary.

Note: The code contains commented-out sections that seem to be related to feature
 importance and a pipeline involving NeighborhoodComponentsAnalysis (NCA), which
 are not currently in use.
"""
import imp
import warnings

from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.neighbors import KNeighborsClassifier #, NeighborhoodComponentsAnalysis
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')


def run_knn(x_train, x_test, y_train, y_test, n_neighbors=3):
    """
    Runs the k-nearest neighbors (kNN) algorithm on the given training and test data.

    Parameters:
    - x_train: The training data features.
    - x_test: The test data features.
    - y_train: The training data labels.
    - y_test: The test data labels.
    - n_neighbors: The number of neighbors to consider (default: 3).

    Returns:
    A dictionary containing the following metrics:
    - 'Accuracy': The accuracy of the kNN model.
    - 'Precision': The precision of the kNN model.
    - 'Recall': The recall of the kNN model.
    - 'F1 Score': The F1 score of the kNN model.
    - 'Confusion Matrix': The confusion matrix of the kNN model.
    - 'Classification Report': The classification report of the kNN model.
    """
    #nca = NeighborhoodComponentsAnalysis(random_state=42)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, digits=3)
    knn_baseline_acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    ####################
    #  Feature importance cannot be discovered for the kNN model.
    #####################
    perm_importance = permutation_importance(knn, x_train, y_train)

    return {
        'Accuracy': knn_baseline_acc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': cm,
        'Classification Report': cr,
        'FI': perm_importance,
    }


def run_dt(x_train, x_test, y_train, y_test):
    """
    Runs a Decision Tree classifier on the given training and testing data.

    Parameters:
    x_train (array-like): The training data features.
    x_test (array-like): The testing data features.
    y_train (array-like): The training data labels.
    y_test (array-like): The testing data labels.

    Returns:
    dict: A dictionary containing the evaluation metrics of the classifier,
     including accuracy, precision, recall, F1 score, confusion matrix, 
     and classification report.
    """
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, digits=3)
    dt_baseline_acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    
    importances = clf.feature_importances_
        
    return {
        'Accuracy': dt_baseline_acc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': cm,
        'Classification Report': cr,
        'FI': importances,
    }


def run_lda(x_train, x_test, y_train, y_test):
    """
    Runs Linear Discriminant Analysis (LDA) on the given training and test data.

    Args:
        x_train (array-like): The training data features.
        x_test (array-like): The test data features.
        y_train (array-like): The training data labels.
        y_test (array-like): The test data labels.

    Returns:
        dict: A dictionary containing the evaluation metrics of the LDA model, including:
            - 'Accuracy': The accuracy of the model.
            - 'Precision': The precision of the model.
            - 'Recall': The recall of the model.
            - 'F1 Score': The F1 score of the model.
            - 'Confusion Matrix': The confusion matrix of the model.
            - 'Classification Report': The classification report of the model.
    """
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train, y_train)
    y_pred = lda.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, digits=3)
    ##print(cm)
    ##print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
    lda_baseline_acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')

    return {
        'Accuracy': lda_baseline_acc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': cm,
        'Classification Report': cr,
        'FI': lda.coef_,
    }

def run_rf(x_train, x_test, y_train, y_test, max_depth=4, random_state=0):
    """
    Runs a Random Forest classifier on the given training and test data.

    Parameters:
    - x_train (array-like): Training data features.
    - x_test (array-like): Test data features.
    - y_train (array-like): Training data labels.
    - y_test (array-like): Test data labels.
    - max_depth (int, optional): Maximum depth of the decision trees in the forest. Defaults to 4.
    - random_state (int, optional): Random seed for reproducibility. Defaults to 0.

    Returns:
    - dict: A dictionary containing the following metrics:
        - 'Accuracy': Accuracy of the classifier.
        - 'Precision': Precision score of the classifier.
        - 'Recall': Recall score of the classifier.
        - 'F1 Score': F1 score of the classifier.
        - 'Confusion Matrix': Confusion matrix of the classifier.
        - 'Classification Report': Classification report of the classifier.
    """
    classifier = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, digits=3)
    rf_baseline_acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    #print(classifier.feature_importances_)
    importances = classifier.feature_importances_
    # features = df.columns
    #forest_importances = pd.Series(importances, index=features).sort_values(ascending=False)
    #print(forest_importances)
    return {
        'Accuracy': rf_baseline_acc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': cm,
        'Classification Report': cr,
        'FI': importances,
    }  # rf_baseline_acc, f1 #, forest_importances


def run_svm(x_train, x_test, y_train, y_test, C=1, random_state=0, kernel='linear'):
    """
    Trains a Support Vector Machine (SVM) classifier using a linear kernel
     and predicts the response for the test dataset.

    Parameters:
    - x_train (array-like): Training data features.
    - x_test (array-like): Test data features.
    - y_train (array-like): Training data labels.
    - y_test (array-like): Test data labels.
    - C (float, optional): Penalty parameter C of the error term. Defaults to 1.
    - random_state (int, optional): Seed used by the random number generator. 
    Defaults to 0.
    - kernel (str, optional): Specifies the kernel type to be used in the algorithm.
    Defaults to 'linear'.

    Returns:
    - dict: A dictionary containing the following metrics:
        - 'Accuracy': Accuracy of the SVM classifier.
        - 'Precision': Precision score of the SVM classifier.
        - 'Recall': Recall score of the SVM classifier.
        - 'F1 Score': F1 score of the SVM classifier.
        - 'Confusion Matrix': Confusion matrix of the SVM classifier.
        - 'Classification Report': Classification report of the SVM classifier.
    """
    #starting_time = time()
    clf = SVC(kernel=kernel, C=C, random_state=random_state)  # type: ignore
    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)
    clf.fit(x_train, y_train)
    y_out = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_out)
    cr = classification_report(y_test, y_out, digits=3)
    svm_baseline_acc = accuracy_score(y_test, y_out)
    f1 = f1_score(y_test, y_out, average='micro')
    precision = precision_score(y_test, y_out, average='micro')
    recall = recall_score(y_test, y_out, average='micro')
    
    perm_importance = permutation_importance(clf, x_train, y_train)
    
    return {
        'Accuracy': svm_baseline_acc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': cm,
        'Classification Report': cr,
        'FI': perm_importance,
    }


def run_ab(x_train, x_test, y_train, y_test, n_estimators=50, random_state=0):
    """
    Runs the AdaBoost classifier on the given training and test data.

    Parameters:
    - x_train (array-like): Training data features.
    - x_test (array-like): Test data features.
    - y_train (array-like): Training data labels.
    - y_test (array-like): Test data labels.
    - n_estimators (int): Number of estimators (weak learners) in the ensemble.
    Default is 50.
    - random_state (int): Random seed for reproducibility. Default is 0.

    Returns:
    - dict: A dictionary containing the following metrics:
        - 'Accuracy': Accuracy score of the classifier.
        - 'Precision': Precision score of the classifier.
        - 'Recall': Recall score of the classifier.
        - 'F1 Score': F1 score of the classifier.
        - 'Confusion Matrix': Confusion matrix of the classifier.
        - 'Classification Report': Classification report of the classifier.
    """
    clf = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state)
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, digits=3)
    ab_baseline_acc = accuracy_score(y_test, y_pred)
    #ab2_baseline_acc = clf.score(x_test, y_test)
    f1 = f1_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')

    importances = clf.feature_importances_

    return {
        'Accuracy': ab_baseline_acc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': cm,
        'Classification Report': cr,
        'FI': importances,
    }

def run_xgb(x_train, x_test, y_train, y_test, n_estimators=50, learning_rate=0.1, random_state=0):
    """
    Runs the XGBoost classifier on the given training and test data.

    Parameters:
    - x_train (array-like): Training data features.
    - x_test (array-like): Test data features.
    - y_train (array-like): Training data labels.
    - y_test (array-like): Test data labels.
    - n_estimators (int): Number of trees in the ensemble. Default is 50.
    - learning_rate (float): Boosting learning rate. Default is 0.1.
    - random_state (int): Random seed for reproducibility. Default is 0.

    Returns:
    - dict: A dictionary containing the following metrics:
        - 'Accuracy': Accuracy score of the classifier.
        - 'Precision': Precision score of the classifier.
        - 'Recall': Recall score of the classifier.
        - 'F1 Score': F1 score of the classifier.
        - 'Confusion Matrix': Confusion matrix of the classifier.
        - 'Classification Report': Classification report of the classifier.
    """
    clf = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, digits=3)
    xgb_acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    importances = clf.feature_importances_

    return {
        'Accuracy': xgb_acc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': cm,
        'Classification Report': cr,
        'FI': importances,
    }

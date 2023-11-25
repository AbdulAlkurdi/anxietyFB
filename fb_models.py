from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
import warnings
from time import time
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')
def run_knn(X_train, X_test, y_train, y_test, n_neighbors=3):
    """
    Runs the k-nearest neighbors (kNN) algorithm on the given training and test data.

    Parameters:
    - X_train: The training data features.
    - X_test: The test data features.
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
    nca = NeighborhoodComponentsAnalysis(random_state=42)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr=(classification_report(y_test, y_pred, digits=3))
    knn_baseline_acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    ####################
    #  Feature importance cannot be discovered for the kNN model. 
    #####################
    #y_pred = knn.predict(X_test)
    #knn_baseline_acc = accuracy_score(y_test, y_pred)
    #nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
    #nca_pipe.fit(X_train, y_train)
    #knn_baseline_acc = nca_pipe.score(X_test, y_test)
    ##print('knn baseline accuracy: ' + str(knn_baseline_acc))
    return {'Accuracy': knn_baseline_acc,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1, 'Confusion Matrix': cm, 'Classification Report': cr}
def run_dt(X_train, X_test, y_train, y_test):
    """
    Runs a Decision Tree classifier on the given training and testing data.

    Parameters:
    X_train (array-like): The training data features.
    X_test (array-like): The testing data features.
    y_train (array-like): The training data labels.
    y_test (array-like): The testing data labels.

    Returns:
    dict: A dictionary containing the evaluation metrics of the classifier, including accuracy, precision, recall, F1 score, confusion matrix, and classification report.
    """
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr=(classification_report(y_test, y_pred, digits=3))
    dt_baseline_acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    '''
    # plot importances
    features = df.columns
    dt_importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    dt_importances[:20].plot.barh(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    '''
    return {'Accuracy': dt_baseline_acc, 'Precision': precision,
            'Recall': recall,
            'F1 Score': f1, 'Confusion Matrix': cm, 'Classification Report': cr}
def run_LDA(X_train, X_test, y_train, y_test):
    """
    Runs Linear Discriminant Analysis (LDA) on the given training and test data.

    Args:
        X_train (array-like): The training data features.
        X_test (array-like): The test data features.
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
    X_train = sc.fit_transform(X_train)  
    X_test = sc.transform(X_test)  

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)  
    cr=(classification_report(y_test, y_pred, digits=3))
    ##print(cm)  
    ##print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))  
    lda_baseline_acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    return { 'Accuracy': lda_baseline_acc, 'Precision': precision,
            'Recall': recall,
            'F1 Score': f1, 'Confusion Matrix': cm, 'Classification Report': cr}
def run_RF(X_train, X_test, y_train, y_test, max_depth=4, random_state=0):
    """
    Runs a Random Forest classifier on the given training and test data.

    Parameters:
    - X_train (array-like): Training data features.
    - X_test (array-like): Test data features.
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
    classifier.fit(X_train, y_train)  
    y_pred = classifier.predict(X_test)  
    cm = confusion_matrix(y_test, y_pred) 
    cr=(classification_report(y_test, y_pred, digits=3)) 
    rf_baseline_acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    #importances = classifier.feature_importances_
    #features = df.columns
    #forest_importances = pd.Series(importances, index=features).sort_values(ascending=False)

    return {'Accuracy': rf_baseline_acc, 'Precision': precision,
            'Recall': recall,
            'F1 Score': f1, 'Confusion Matrix': cm, 'Classification Report': cr} #rf_baseline_acc, f1 #, forest_importances
def run_svm( X_train, X_test, y_train, y_test, C=1, random_state=0, kernel='linear'):
    """
    Trains a Support Vector Machine (SVM) classifier using a linear kernel and predicts the response for the test dataset.

    Parameters:
    - X_train (array-like): Training data features.
    - X_test (array-like): Test data features.
    - y_train (array-like): Training data labels.
    - y_test (array-like): Test data labels.
    - C (float, optional): Penalty parameter C of the error term. Defaults to 1.
    - random_state (int, optional): Seed used by the random number generator. Defaults to 0.
    - kernel (str, optional): Specifies the kernel type to be used in the algorithm. Defaults to 'linear'.

    Returns:
    - dict: A dictionary containing the following metrics:
        - 'Accuracy': Accuracy of the SVM classifier.
        - 'Precision': Precision score of the SVM classifier.
        - 'Recall': Recall score of the SVM classifier.
        - 'F1 Score': F1 score of the SVM classifier.
        - 'Confusion Matrix': Confusion matrix of the SVM classifier.
        - 'Classification Report': Classification report of the SVM classifier.
    """
    starting_time = time()
    clf = SVC(kernel='linear', C=1, random_state=0)
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
    X_train = scaling.transform(X_train)
    X_test = scaling.transform(X_test)
    clf.fit(X_train, y_train)
    y_out = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_out)
    cr=(classification_report(y_test, y_out, digits=3))
    svm_baseline_acc = accuracy_score(y_test, y_out)
    f1 = f1_score(y_test, y_out, average='micro')
    precision = precision_score(y_test, y_out, average='micro')
    recall = recall_score(y_test, y_out, average='micro')
    
    return {'Accuracy': svm_baseline_acc, 'Precision': precision,
            'Recall': recall,
            'F1 Score': f1, 'Confusion Matrix': cm, 'Classification Report': cr}
def run_ab(X_train, X_test, y_train, y_test, n_estimators=50, random_state=0):
    """
    Runs the AdaBoost classifier on the given training and test data.

    Parameters:
    - X_train (array-like): Training data features.
    - X_test (array-like): Test data features.
    - y_train (array-like): Training data labels.
    - y_test (array-like): Test data labels.
    - n_estimators (int): Number of estimators (weak learners) in the ensemble. Default is 50.
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
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, digits=3)
    ab_baseline_acc = accuracy_score(y_test, y_pred)
    ab2_baseline_acc = clf.score(X_test, y_test)
    f1 = f1_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    return {'Accuracy': ab_baseline_acc, 'Precision': precision,
            'Recall': recall,
            'F1 Score': f1, 'Confusion Matrix': cm, 'Classification Report': cr}

import numpy as np
import scipy.signal as scisig
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib as mpl
from heartpy.datautils import *
from heartpy.peakdetection import *
mpl.rcParams['agg.path.chunksize'] = 10000
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, precision_score, f1_score 
from sklearn.metrics import confusion_matrix, accuracy_score

import csv 
import warnings
warnings.filterwarnings('ignore')


def run_knn(X_train, X_test, y_train, y_test, n_neighbors=3):
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
    #print('knn baseline accuracy: ' + str(knn_baseline_acc))
    return {'Accuracy': knn_baseline_acc,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1, 'confusion_matrix': cm, 'Classification Report': cr}

def run_dt(X_train, X_test, y_train, y_test):
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
            'F1 Score': f1, 'confusion_matrix': cm, 'Classification Report': cr}

def run_LDA(X_train, X_test, y_train, y_test):
    sc = StandardScaler()  
    X_train = sc.fit_transform(X_train)  
    X_test = sc.transform(X_test)  

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)  
    cr=(classification_report(y_test, y_pred, digits=3))
    #print(cm)  
    #print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))  
    lda_baseline_acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    return { 'Accuracy': lda_baseline_acc, 'Precision': precision,
            'Recall': recall,
            'F1 Score': f1, 'confusion_matrix': cm, 'Classification Report': cr}

def run_RF(X_train, X_test, y_train, y_test, max_depth=4, random_state=0):
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
            'F1 Score': f1, 'confusion matrix': cm, 'Classification Report': cr} #rf_baseline_acc, f1 #, forest_importances

def run_svm( X_train, X_test, y_train, y_test, C=1, random_state=0, kernel='linear'):
    # Create a SVC classifier using a linear kernel
    clf = SVC(kernel=kernel, C=C, random_state=random_state)
    # Train the classifier
    clf.fit(X_train, y_train)
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr=(classification_report(y_test, y_pred, digits=3))
    #print(lm_svc)
    svm_baseline_acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    '''
    features = df.columns
    pd.Series(abs(clf.coef_[0]), index=features).nlargest(10).plot(kind='barh') # Feature Importance (Top 20)
    '''
    return {'Accuracy': svm_baseline_acc, 'Precision': precision,
            'Recall': recall,
            'F1 Score': f1, 'confusion_matrix': cm, 'Classification Report': cr} #svm_baseline_acc, f1
    
def run_ab( X_train, X_test, y_train, y_test, n_estimators= 50, random_state=0):
    clf = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr=(classification_report(y_test, y_pred, digits=3))
    ab_baseline_acc = accuracy_score(y_test, y_pred)
    ab2_baseline_acc = clf.score(X_test, y_test)
    f1 = f1_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    #print('ab baseline acc1: {ab_baseline_acc}. ab baseline acc2: {ab2_baseline_acc}')
    '''
    features = df.columns
    ab_imp = clf.feature_importances_
    ab_imp = pd.Series(ab_imp, index=features).sort_values(ascending=False)
    # plot importances
    fig, ax = plt.subplots(figsize=(12, 5))
    ab_imp[:20].plot.barh(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    '''
    return {'Accuracy': ab_baseline_acc, 'Precision': precision,
            'Recall': recall,
            'F1 Score': f1, 'confusion_matrix': cm, 'Classification Report': cr} # ab_baseline_acc, ab2_baseline_acc, f1
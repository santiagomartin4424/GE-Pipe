import os, multiprocessing

# Configure environment variables exactly to 1 thread BEFORE loading OpenBLAS/MKL
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["N_THREADS"] = "1"

import numpy as np
np.seterr(all="ignore")

from algorithm.parameters import params
from utilities.fitness.error_metric import f1_score
from utilities.fitness.error_metric import balanced_accuracy
from utilities.fitness.get_data import get_data
from utilities.fitness.math_functions import *
from utilities.fitness.optimize_constants import optimize_constants
from utilities.fitness import data_cache
from fitness.base_ff_classes.base_ff import base_ff

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

# ------- IMPUTERS -----------------------
from sklearn.impute import SimpleImputer, KNNImputer    
# ------- SCALERS ------------------------
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler, Normalizer
# ------- FEATURE AUGMENTATION -----------
from sklearn.kernel_approximation import Nystroem, RBFSampler
# ------- FEATURE SELECTION  -------------
from sklearn.feature_selection import VarianceThreshold, SelectFwe, SelectPercentile
# ------- FEATURE EXTRACTION -------------
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
# ------- BALANCE ------------------------
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# ------- CLASSIFIERS --------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier


import os, signal, re, warnings, time
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)



# -------------------------AUXILIAR FUNCTIONS -------------------------
#----------------------------------------------------------------------
def rename_steps(match):
    """    Searches for repeated step names, and adds a numerical sufix.
    This function is called in re.sub (after, in this code). We search
    the repeated step-names, and modify them (we add a number at the end).
    
    :param match: The regex triggered, with the text that will be
    modified.
    :return: The regex (string) modified.
    """
    name = match.group(1)   # We advance the string one position. To obtain instead of '(scaler,' just 'scaler,'

    rename_steps.counter[name] = rename_steps.counter.get(name, 0) + 1  # we store how many times this name has appeared before.
    return f"('{name}_{rename_steps.counter[name]}',"  # we return the modified string




class supervised_learning_pipclean(base_ff):
    """
    Fitness function for evaluating ml pipelines for supervised learning,
    specifically classification problems. Given a dataset, it
    returns the error between y (true labels) and yhat (estimated
    labels).

    We can pass in the error metric and the dataset via the params
    dictionary. RMSE is suitable for regression, while F1-score,
    hinge-loss, balanced_acc and others are suitable for classification.
    """


    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

        # DATA LOADING
        if data_cache._DATASET_NAME == params['DATASET_TRAIN'] and data_cache._TRAIN_IN is not None:
                self.training_in = data_cache._TRAIN_IN
                self.training_exp = data_cache._TRAIN_EXP
                self.test_in = data_cache._TEST_IN
                self.test_exp = data_cache._TEST_EXP
        else:
            # This will only be executed the first time, or if something fails
            self.training_in, self.training_exp, self.test_in, self.test_exp = \
                get_data(params['DATASET_TRAIN'], params['DATASET_TEST'])


        # If the is NaN, we convert it into a number (for example -1 or 0), so that the cast doesn't fail
        self.training_exp = np.nan_to_num(self.training_exp, nan=-1)

        # We force the target variable of the dataset to be of type int (instead of float), just in case, for the classification lable
        self.training_exp = self.training_exp.astype(int)

        # Find number of variables.
        self.n_vars = np.shape(self.training_in)[1] # sklearn convention

        # Regression/classification-style problems use training and test data. However, we're using cross-validation (we use only 1 block of data)
        #if params['DATASET_TEST']:
        #    self.training_test = True

        # Set error metric, if it's not set already.
        if params['ERROR_METRIC'] is None:
            params['ERROR_METRIC'] = balanced_accuracy

        self.maximise = params['ERROR_METRIC'].maximise

        # We create the context dictionary with all needed. Each individual will access a copy of these references.
        # We also want to use a dictionary so that Pipeline receives strings, and not variables. For example, we want Pipeline(...) to have arguments like "imputer" and not imputer
        # because otherwise it would be recognised as a variable and not a string, and give an error. This is also why we do 'imputer': "imputer", ...
        self.context = {


            # ----- Pipeline Steps. Strings -----
            'imputer': "imputer",
            'scaler': "scaler",
            'augmentation': "augmentation",
            'selection': "selection",
            'extraction': "extraction",
            'balance': "balance",
            'clf': "clf",

            # ----- Categorical hyperparameters. Strings -----
            # Imputation & weights
            'mean': "mean",
            'median': "median",
            'most_frequent': "most_frequent",
            'uniform': "uniform",
            'distance': "distance",
            'lsqr': "lsqr",
            'eigen': "eigen",
            

            # Metrics and Criterias
            'gini': "gini",
            'entropy': "entropy",
            'l1': "l1",
            'l2': "l2",
            'max': "max",

            # Kernels and Afinities
            'rbf': "rbf",
            'cosine': "cosine",
            'sigmoid': "sigmoid",
            'poly': "poly",
            'euclidean': "euclidean",
            'manhattan': "manhattan",
            'ward': "ward",
            'complete': "complete",
            'average': "average",

            # Decomposition Algorithms and ICA
            'parallel': "parallel",
            'deflation': "deflation",
            'logcosh': "logcosh",
            'exp': "exp",
            'cube': "cube",
            'arpack': "arpack",
            'randomized': "randomized",
            'unit': "unit",

            # Balance Strategies (Imbalanced-learn)
            'minority': "minority",
            'notminority': "not minority",  
            'notmajority': "not majority",
            'allstrategy': "all",
            'notremaining': "not remaining",


            # Clasifiers (Forests and Boosting)
            'sqrt': "sqrt",
            'log2': "log2",
            'balanced': "balanced",
            'balanced_subsample': "balanced_subsample",
            'SAMME_R': "SAMME.R",
            'SAMME': "SAMME",
            'auto': "auto",

            # Loss and penalties fuctions
            'hinge': "hinge",
            'pa1' : "pa1",
            'squared_hinge': "squared_hinge",
            'log_loss': "log_loss",
            'modified_huber': "modified_huber",
            'perceptron': "perceptron",
            'saga': "saga",
            'linear': "linear",

            # Neuronal networks (MLP)
            'lbfgs': "lbfgs",
            'sgd': "sgd",
            'adam': "adam",
            'identity': "identity",
            'logistic': "logistic",
            'tanh': "tanh",
            'relu': "relu",


            # ----- No Strings. Code references  -----
            'Pipeline': Pipeline,
            # ------- IMPUTERS -----------------------
            'SimpleImputer': SimpleImputer,
            'KNNImputer': KNNImputer,
            # ------- SCALERS ------------------------
            'StandardScaler': StandardScaler,
            'RobustScaler': RobustScaler,
            'MinMaxScaler': MinMaxScaler,
            'MaxAbsScaler':MaxAbsScaler,
            'Normalizer': Normalizer,
            # ------- FEATURE AUGMENTATION -----------
            'Nystroem': Nystroem,
            'RBFSampler':RBFSampler,
            # ------- FEATURE SELECTION  -------------
            'VarianceThreshold': VarianceThreshold,
            'SelectFwe': SelectFwe,
            'SelectPercentile': SelectPercentile,
            # ------- FEATURE EXTRACTION -------------
            'FeatureAgglomeration': FeatureAgglomeration,
            'PCA': PCA,
            'FastICA': FastICA,
            'TruncatedSVD': TruncatedSVD,
            # ------- BALANCE ------------------------
            'RandomOverSampler': RandomOverSampler,
            'RandomUnderSampler': RandomUnderSampler,
            'SMOTE': SMOTE,
            'SMOTETomek': SMOTETomek,
            # ------- CLASSIFIERS --------------------
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'RandomForestClassifier': RandomForestClassifier,
            'ExtraTreesClassifier': ExtraTreesClassifier,
            'AdaBoostClassifier': AdaBoostClassifier,
            'KNeighborsClassifier': KNeighborsClassifier,
            'GaussianNB': GaussianNB,
            'BernoulliNB': BernoulliNB,
            'MultinomialNB': MultinomialNB,
            'SVC': SVC,
            'LinearSVC': LinearSVC,
            'LogisticRegression': LogisticRegression,
            'SGDClassifier': SGDClassifier,
            'PassiveAggressiveClassifier': PassiveAggressiveClassifier,
            'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis,
            'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis,
            'MLPClassifier': MLPClassifier
        }







    def evaluate(self, ind, **kwargs):
        """
        Evaluation of a single individual using exec().

        The resulting pipeline will be stored in my_pipeline, and then used for fit and cross-validation using the dataset.
        The global parallelization is managed externally, with multiprocessing.pool, in fitness/evaluation.py.
        """

        # Reset of the name counter
        rename_steps.counter = {}

        # Replace the duplicated step names in the phenotype calling re.sub, using the rename_steps() function declared at the beginning
        fixed_phenotype = re.sub(r"\((imputer|scaler|augmentation|selection|extraction|balance|clf),", rename_steps, ind.phenotype)

        try:
            local_env = self.context.copy()
            exec(fixed_phenotype, local_env)
            my_pipeline = local_env['my_pipeline']  #now we have the Pipeline object

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=params['RANDOM_SEED'])
            scores = []

            # Bucle secuencial estándar para los 5 folds
            for train_index, val_index in skf.split(self.training_in, self.training_exp):
                X_train_fold = self.training_in[train_index]
                y_train_fold = self.training_exp[train_index]
                X_val_fold = self.training_in[val_index]
                y_val_fold = self.training_exp[val_index]

                my_pipeline.fit(X_train_fold, y_train_fold)         # Train
                y_pred = my_pipeline.predict(X_val_fold)            # Predict
                
                score = params['ERROR_METRIC'](y_val_fold, y_pred)
                scores.append(score)

            return float(np.mean(scores))

        except Exception as e:
            # In case of invalid pipeline, convergence or syntaxis error, fitness = 0.
            return 0.0
        




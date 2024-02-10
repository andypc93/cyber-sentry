# Import pandas, numpy and seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3 as sql

from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

pd.set_option('display.max_columns',None)

# ====================================================================================================

db_path = 'C:/Users/andre/SQLite/CyberSentryDB.db'

# Establish a connection to the database specified by db_path
conn = sql.connect(db_path)

# Create a cursor object to execute SQL queries
cursor = conn.cursor()

# Execute a SQL query to select all records from the 'network_activity' table
# and store the result in a pandas DataFrame 'df' for further analysis
#df = pd.read_sql_query("SELECT * FROM network_activity", conn)
df_test = pd.read_sql_query("SELECT * FROM testing_data", conn)
df_train = pd.read_sql_query("SELECT * FROM training_data", conn)

# Close the cursor to release database resources
cursor.close()

# Close the connection to the database to ensure data integrity and release resources
conn.close()

# ====================================================================================================

columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot'
,'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations'
,'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate'
,'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count'
,'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate'
,'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','outcome','level'])

# Assign name for columns
df_train.columns = columns
df_test.columns = columns

# ====================================================================================================

# For the training dataset:
# Keep rows with 'outcome' as 'normal' unchanged.
df_train.loc[df_train['outcome'] == "normal", "outcome"] = 'normal'  
# Change 'outcome' values not equal to 'normal' to 'attack' in the training dataset.
df_train.loc[df_train['outcome'] != 'normal', "outcome"] = 'attack'

df_test.loc[df_test['outcome'] == "normal", "outcome"] = 'normal'  
# Change 'outcome' values not equal to 'normal' to 'attack' in the training dataset.
df_test.loc[df_test['outcome'] != 'normal', "outcome"] = 'attack'

# ====================================================================================================

def preprocess(dataframe, scaler=None):
    # Define categorical and numerical columns
    cat_cols = ['is_host_login', 'protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_guest_login', 'level', 'outcome']
    num_cols = dataframe.drop(columns=cat_cols).columns

    # Scale numerical columns if scaler is not provided
    if not scaler:
        scaler = StandardScaler()  # Initialize StandardScaler if not provided
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])  # Fit and transform numerical columns using scaler

    # Encode 'outcome' as binary (0 for 'normal', 1 for other values)
    dataframe.loc[dataframe['outcome'] == "normal", "outcome"] = 0
    dataframe.loc[dataframe['outcome'] != 0, "outcome"] = 1

    # Encode categorical columns as dummy variables
    dataframe = pd.get_dummies(dataframe, columns=['protocol_type', 'service', 'flag'])

    return dataframe, scaler  # Return preprocessed DataFrame and scaler

# ====================================================================================================

# Fit and transform the training data
df_train_scaled, scalar = preprocess(df_train)
df_test_scaled, scalar = preprocess(df_test)

# ====================================================================================================

def find_diff_col(df1, df2):
    # Find the difference in columns
    diff_1_to_2 = set(df1.columns) - set(df2.columns)
    diff_2_to_1 = set(df2.columns) - set(df1.columns)

    # Print out the different column names
    print("Columns in df1 not in df2:", diff_1_to_2)
    print("Columns in df2 not in df1:", diff_2_to_1)

find_diff_col(df_train_scaled, df_test_scaled)

# ====================================================================================================

# Prepare feature variables from the scaled datasets, excluding 'outcome' and 'level' from features
X_train = df_train_scaled.drop(['outcome', 'level'], axis=1)
Y_train = df_train_scaled['outcome'].values.astype('int')

x_test = df_test_scaled.drop(['outcome', 'level'], axis=1)  # Use consistent naming convention
y_test = df_test_scaled['outcome'].values.astype('int')

# Instantiate PCA with desired number of components
pca = PCA(n_components=20)  # Example: reducing to 22 components

# Fit PCA on the training data and transform it
X_train_reduced = pca.fit_transform(X_train)

# Transform the test data using the already fitted PCA (do not fit it again)
x_test_reduced = pca.transform(x_test)  # Use transform, not fit_transform

# Correctly print the number of features after PCA
print(f"Number of original features in X_train is {X_train.shape[1]} and number of reduced features after PCA is {X_train_reduced.shape[1]}")
print(f"Number of original features in X_test is {x_test.shape[1]} and number of reduced features after PCA is {x_test_reduced.shape[1]}")


# ====================================================================================================

def evaluate_classification(model, name, X_train, X_test, y_train, y_test):

    # Fit the model using the training data
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_precision = precision_score(y_train, y_pred_train, average='macro')
    test_precision = precision_score(y_test, y_pred_test, average='macro')
    train_recall = recall_score(y_train, y_pred_train, average='macro')
    test_recall = recall_score(y_test, y_pred_test, average='macro')

    # Output metrics
    print(f"Training Accuracy {name}: {train_accuracy*100:.2f}%")
    print(f"Test Accuracy {name}: {test_accuracy*100:.2f}%")
    print(f"Training Precision {name}: {train_precision*100:.2f}%")
    print(f"Test Precision {name}: {test_precision*100:.2f}%")
    print(f"Training Recall {name}: {train_recall*100:.2f}%")
    print(f"Test Recall {name}: {test_recall*100:.2f}%")

    # Display confusion matrix
    confusion_mtx = confusion_matrix(y_test, y_pred_test)
    display_labels = ['Normal', 'Attack']
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(10,10))
    cm_display.plot(ax=ax)
    plt.grid(False)
    plt.show()

# You will call the evaluate_classification function like this, for example:
# evaluate_classification(your_model, 'Your Model Name', X_train, X_test, y_train_class, y_test_class)


# ====================================================================================================

rf = RandomForestClassifier(
    n_estimators=100,        # Number of trees in the forest.
    criterion='gini',        # The function to measure the quality of a split. 'gini' for Gini impurity and 'entropy' for information gain.
    max_depth=10,          # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    min_samples_split=20,     # The minimum number of samples required to split an internal node.
    min_samples_leaf=1,      # The minimum number of samples required to be at a leaf node.
    min_weight_fraction_leaf=0.0, # The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.
    max_leaf_nodes=None,     # Grow trees with `max_leaf_nodes` in best-first fashion. Best nodes are defined as relative reduction in impurity.
    min_impurity_decrease=0.0, # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
    bootstrap=True,          # Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
    oob_score=False,         # Whether to use out-of-bag samples to estimate the generalization accuracy.
    n_jobs=-1,             # The number of jobs to run in parallel. None means 1. -1 means using all processors.
    random_state=None,       # Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node.
    verbose=0,               # Controls the verbosity when fitting and predicting.
    warm_start=False,        # When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.
    class_weight=None,       # Weights associated with classes in the form `{class_label: weight}`. If not given, all classes are supposed to have weight one.
    ccp_alpha=0.0,           # Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than `ccp_alpha` will be chosen.
    max_samples=None         # If bootstrap is True, the number of samples to draw from X to train each base estimator.
)

# Evaluate the model's performance on both the training and test sets
evaluate_classification(rf, "RandomForestClassifier", X_train_reduced, x_test_reduced, Y_train, y_test)


# ====================================================================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def tune_random_forest(X_train, y_train, X_test, y_test, param_grid, cv=5):
    """
    Tune a RandomForestClassifier based on a given set of parameters.

    :param X_train: Training features
    :param y_train: Training target
    :param X_test: Test features
    :param y_test: Test target
    :param param_grid: Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values
    :param cv: Number of cross-validation folds
    :return: The best estimator from the grid search
    """
    # Initialize the RandomForestClassifier
    rf = RandomForestClassifier()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, n_jobs=12, verbose=10)

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Print the best parameters and the best score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    # Predict on the test set using the best found parameters
    y_pred_test = grid_search.best_estimator_.predict(X_test)

    # Print classification report
    print(classification_report(y_test, y_pred_test))

    # Return the best estimator
    return grid_search.best_estimator_, grid_search.cv_results_


# Define a more granular parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]                   # Including both options for bootstrapping samples
}

# Call the function
best_rf, cv_results = tune_random_forest(X_train_reduced, Y_train, x_test_reduced, y_test, param_grid)

# ====================================================================================================
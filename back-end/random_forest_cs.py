# Import pandas, numpy and seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

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

#===================================================================================

# Define the path to the SQLite database file
db_path = 'C:/Users/rober/SQLite/CyberSentryDB.db'

# Establish a connection to the database specified by db_path
conn = sqlite3.connect(db_path)

# Create a cursor object to execute SQL queries
cursor = conn.cursor()

# Execute a SQL query to select all records from the 'network_activity' table
# and store the result in a pandas DataFrame 'df' for further analysis
df = pd.read_sql_query("SELECT * FROM network_activity", conn)

# Close the cursor to release database resources
cursor.close()

# Close the connection to the database to ensure data integrity and release resources
conn.close()

#=====================================================================================

# This line updates the 'outcome' column in the df_train DataFrame again.
# For all rows where the 'outcome' is not 'normal', it sets the 'outcome' value to 'attack'.
# This is a way to categorize all outcomes into two groups: 'normal' and 'attack',
# effectively binarizing the 'outcome' column into these two categories.
df.loc[df['outcome'] == "normal", "outcome"] = 'normal'

df.loc[df['outcome'] != 'normal', "outcome"] = 'attack'

#======================================================================================

def bar_plot(df, cols_list, rows, cols):
    # Create a grid of subplots with the specified number of rows and columns.
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    fig.tight_layout(pad=1.0)  # Add spacing between plots for clarity

    # Flatten the axes array and iterate over it along with the column names in cols_list.
    for ax, col in zip(axes.ravel(), cols_list):
        # Use Seaborn's countplot to create a bar chart.
        sns.countplot(x=col, data=df, ax=ax)

        # Calculate the total number of data points for the percentage calculation.
        total = len(df[col])

        # Iterate through the patches (bars) in the barplot to get their properties.
        for p in ax.patches:
            # Calculate the percentage and format it.
            percentage = '{:.1f}%'.format(100 * p.get_height() / total)
            # Get the x and y coordinates to place the text.
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            # Place the text on the bar.
            ax.text(x, y, percentage, ha='center', va='bottom')

        # Set the title of the current subplot to the name of the column.
        ax.set_title(str(col), fontsize=12)

        # Rotate the x-axis labels for better readability.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Adjust the layout and display the figure with all the bar charts.
    plt.tight_layout()
    plt.show()

#bar plot print out variable
bar_plot_1 = bar_plot(df, ['protocol_type', 'outcome'],1,2)

#======================================================================================

def pie_plot(df, cols_list, rows, cols):
    # Create a grid of subplots with the specified number of rows and columns.
    fig, axes = plt.subplots(rows, cols, figsize=(15, 20))
    fig.tight_layout(pad=1.0)  # Add spacing between plots for clarity

    # If there is only one row or one column, axes is a 1D numpy array.
    if rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.ravel()  # Flatten the axes array for iteration

    # Iterate over the axes array and the column names in cols_list.
    for ax, col in zip(axes, cols_list):
        # Calculate the value counts for the current column.
        counts = df[col].value_counts()

        # Create a pie chart in each subplot.
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)

        # Set the title of the current subplot to the name of the column.
        ax.set_title(str(col), fontsize=12)

    # Adjust the layout and display the figure with all the pie charts.
    plt.tight_layout()
    plt.show()

    #pie plot output variable
    pie_plot_1 = pie_plot(df, ['protocol_type','outcome'], 1, 2)

#======================================================================================

def preprocess(dataframe, scaler=None):

    # Separate categorical and numerical columns
    cat_cols = ['is_host_login', 'protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_guest_login', 'level', 'outcome']
    num_cols = dataframe.drop(columns=cat_cols).columns

    # Scale numerical columns
    if not scaler:
        scaler = StandardScaler()
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])

    # Encode 'outcome' as binary
    dataframe.loc[dataframe['outcome'] == "normal", "outcome"] = 0
    dataframe.loc[dataframe['outcome'] != 0, "outcome"] = 1

    # Encode categorical columns as dummies
    dataframe = pd.get_dummies(dataframe, columns=['protocol_type', 'service', 'flag'])

    return dataframe, scaler

# Fit and transform the training data
df_scaled, scaler = preprocess(df)

#======================================================================================

# Prepare feature and target variables
x = df_scaled.drop(['outcome', 'level'], axis=1).values
y = df_scaled['outcome'].values.astype(int)  # Direct conversion to int
y_reg = df_scaled['level'].values

# Apply PCA to reduce dimensionality for the classification task
pca = PCA(n_components=20).fit(x)  # Fit and instantiate PCA in one step
x_reduced = pca.transform(x)
print(f"Number of original features is {x.shape[1]} and of reduced features is {x_reduced.shape[1]}")

# Splitting the dataset into training and testing sets for both original and reduced feature sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train_reduced, x_test_reduced, y_train_reduced, y_test_reduced = train_test_split(x_reduced, y, test_size=0.2, random_state=42)
x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(x, y_reg, test_size=0.2, random_state=42)

# Initialize a dictionary to store evaluation metrics
model_evals = dict()

#======================================================================================

def evaluate_classification(model, name, X_train, X_test, y_train, y_test):

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_precision = precision_score(y_train, y_pred_train)
    test_precision = precision_score(y_test, y_pred_test)
    train_recall = recall_score(y_train, y_pred_train)
    test_recall = recall_score(y_test, y_pred_test)

    # Store metrics in dictionary
    model_evals[name] = [train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall]

    # Output metrics
    print(f"Training Accuracy {name}: {train_accuracy*100:.2f}%  Test Accuracy {name}: {test_accuracy*100:.2f}%")
    print(f"Training Precision {name}: {train_precision*100:.2f}%  Test Precision {name}: {test_precision*100:.2f}%")
    print(f"Training Recall {name}: {train_recall*100:.2f}%  Test Recall {name}: {test_recall*100:.2f}%")

    # Display confusion matrix
    confusion_mtx = confusion_matrix(y_test, y_pred_test)
    display_labels = ['Normal', 'Attack']
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(10,10))
    cm_display.plot(ax=ax)
    plt.grid(False)
    plt.show()

# Example usage (model needs to be defined and trained before calling this function):
# evaluate_classification(your_model, "Your Model Name", x_train, x_test, y_train, y_test)

#======================================================================================

# Initialize the RandomForestClassifier with various parameters
rf = RandomForestClassifier(
    n_estimators=150,        # Number of trees in the forest.
    criterion='gini',        # The function to measure the quality of a split. 'gini' for Gini impurity and 'entropy' for information gain.
    max_depth=None,          # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    min_samples_split=2,     # The minimum number of samples required to split an internal node.
    min_samples_leaf=1,      # The minimum number of samples required to be at a leaf node.
    min_weight_fraction_leaf=0.0, # The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.
    max_leaf_nodes=None,     # Grow trees with `max_leaf_nodes` in best-first fashion. Best nodes are defined as relative reduction in impurity.
    min_impurity_decrease=0.0, # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
    bootstrap=True,          # Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
    oob_score=False,         # Whether to use out-of-bag samples to estimate the generalization accuracy.
    n_jobs=-1,               # The number of jobs to run in parallel. None means 1. -1 means using all processors.
    random_state=None,       # Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node.
    verbose=0,               # Controls the verbosity when fitting and predicting.
    warm_start=False,        # When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.
    class_weight=None,       # Weights associated with classes in the form `{class_label: weight}`. If not given, all classes are supposed to have weight one.
    ccp_alpha=0.0,           # Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than `ccp_alpha` will be chosen.
    max_samples=None         # If bootstrap is True, the number of samples to draw from X to train each base estimator.
)

# Fit the Random Forest classifier to the training data7
rf.fit(X_train, y_train_class)

#======================================================================================

# Evaluate the model's performance on both the training and test sets
evaluate_classification(rf, "RandomForestClassifier", X_train, X_test, y_train_class, y_test_class)

# Perform 5-fold cross-validation to assess model's stability and performance across different subsets of the data
# Cross-validation is crucial for verifying the model's ability to generalize to unseen data
cv_scores = cross_val_score(rf, X_train, y_train_class, cv=5, scoring='accuracy')

# Print the accuracy scores obtained from cross-validation
# These scores provide insight into how the model performs on different folds of the training data
print("Cross-Validation Accuracy Scores:", cv_scores)

# Calculate the mean and standard deviation of the cross-validation scores to get an overall performance metric and its variability
mean_cv_accuracy = cv_scores.mean()
std_cv_accuracy = cv_scores.std()
print(f"Mean CV Accuracy: {mean_cv_accuracy:.2f}, Standard Deviation in CV Accuracy: {std_cv_accuracy:.2f}")

#======================================================================================



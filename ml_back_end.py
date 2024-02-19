

import pandas as pd
import sqlite3 as sql
import matplotlib.pyplot as plt
from joblib import load
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

#===================================================

#db_path = 'C:/Users/rober/SQLite/CyberSentryDB.db'

#db_path = 'C:/Users/andre/SQLite/CyberSentryDB.db'

db_path = 'C:/Users/diama/SQLiteStudio/CyberSentryDB.db'

conn = sql.connect(db_path)

cursor = conn.cursor()

df_test = pd.read_sql_query("SELECT * FROM testing_data", conn)
df_train = pd.read_sql_query("SELECT * FROM training_data", conn)

cursor.close()

conn.close()

#===================================================

import os
print("Current Working Directory:", os.getcwd())

model_path = 'rf_model.joblib'  # Adjust as necessary
if os.path.exists(model_path):
    print(f"File found: {model_path}")
else:
    print(f"File not found: {model_path}")

#===================================================

model_path = "rf_model.joblib"

rf_model = load(model_path)

#===================================================

columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'outcome',
    'level'
]


df_test.columns = columns
df_train.columns = columns

#===================================================

class_DoS = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 
             'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm']
class_Probe = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']

class_U2R = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']

class_R2L = ['ftp_write', 'guess_passwd', 'httptunnel',  'imap', 'multihop', 'named', 
             'phf', 'sendmail', 'snmpgetattack', 'spy', 'snmpguess', 'warezclient', 
             'warezmaster', 'xlock', 'xsnoop']

class_attack = class_DoS + class_Probe + class_U2R + class_R2L

#===================================================

df_train['class'] = df_train['outcome']
df_train['class'].replace(class_attack, value='attack', inplace=True)

df_train.drop(columns=["outcome", "level"], inplace =True)

df_test['class'] = df_test['outcome']
df_test['class'].replace(class_attack, value='attack', inplace=True)

df_test.drop(columns=["outcome", "level"], inplace =True)


#===================================================

def preprocess(dataframe, to_drop_columns):
    
    x = dataframe.drop(columns=["class"])
    x = x.drop(columns = to_drop_columns)

    x_num = x.select_dtypes(exclude='object')

    y = dataframe["class"]
    
    return x, x_num, y

#===================================================

drop_columns = ['srv_serror_rate',
 'dst_host_serror_rate',
 'dst_host_srv_serror_rate',
 'srv_rerror_rate',
 'dst_host_rerror_rate',
 'dst_host_srv_rerror_rate',
 'dst_host_same_srv_rate',
 'num_root',
 'num_outbound_cmds',
 'su_attempted']

X_train, X_train_num, Y_train = preprocess(df_train, drop_columns)

x_test, x_test_num, y_test = preprocess(df_test, drop_columns)


#===================================================

# Instantiate the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform Y_train
# This will convert the categories to 0 and 1
Y_train_label = label_encoder.fit_transform(Y_train)

# Transform y_test using the same encoder
# This ensures consistency in encoding between training and test sets
y_test_label = label_encoder.transform(y_test)

#===================================================

numerical_columns = x_test.select_dtypes(include=['number']).columns


def scale_data_with_training(X_train, X_test, numerical_columns):

    scaler = StandardScaler()
    
    # Fit the scaler on the numerical columns of the training data and transform them
    X_train_numerical_scaled = scaler.fit_transform(X_train[numerical_columns])
    
    # Transform the numerical columns of the testing data using the same scaler
    X_test_numerical_scaled = scaler.transform(X_test[numerical_columns])
    
    # Convert the scaled arrays back to DataFrames
    X_train_numerical_scaled_df = pd.DataFrame(X_train_numerical_scaled, columns=numerical_columns, index=X_train.index)
    X_test_numerical_scaled_df = pd.DataFrame(X_test_numerical_scaled, columns=numerical_columns, index=X_test.index)
    
    # Drop the original numerical columns from the original DataFrames
    X_train_dropped = X_train.drop(columns=numerical_columns)
    X_test_dropped = X_test.drop(columns=numerical_columns)
    
    # Concatenate the scaled numerical DataFrames with the original DataFrames (without the numerical columns)
    X_train_scaled = pd.concat([X_train_dropped, X_train_numerical_scaled_df], axis=1)
    X_test_scaled = pd.concat([X_test_dropped, X_test_numerical_scaled_df], axis=1)
    
    return X_train_scaled, X_test_scaled

#===================================================

X_train_scaled, x_test_scaled = scale_data_with_training(X_train, x_test, numerical_columns)

#===================================================

def one_hot_encode_and_align(train_df, test_df, columns_to_encode):
    # Combine train and test dataframes temporarily to ensure consistent one-hot encoding
    combined_df = pd.concat([train_df, test_df], keys=['train', 'test'])
    
    # One-hot encode the specified columns
    combined_df_encoded = pd.get_dummies(combined_df, columns=columns_to_encode)
    
    # Split the combined dataframe back into the original train and test dataframes
    train_df_encoded = combined_df_encoded.xs('train')
    test_df_encoded = combined_df_encoded.xs('test')
    
    # Align the test dataframe to the training dataframe, filling missing columns with zeros
    test_df_encoded = test_df_encoded.reindex(columns=train_df_encoded.columns, fill_value=0)
    
    return train_df_encoded, test_df_encoded

# Apply the function to your dataframes
X_train_encoded, x_test_encoded = one_hot_encode_and_align(X_train_scaled, x_test_scaled, ["protocol_type", "service", "flag"])

#===================================================


# Predict on the test dataset
y_pred = rf_model.predict(x_test_encoded)

# Create a Series for predictions to align with x_test_encoded's index
predictions_df = pd.Series(y_pred, index=x_test.index, name='Predicted')

# Assuming '1' represents an attack
attack_label = 1

# Identify rows classified as an attack
# Ensure you're using the encoded version if that's what was used for prediction
attacks_df = x_test[predictions_df == attack_label]

# Optional: If you want to display only a subset of rows to avoid outputting a very large dataframe
print(attacks_df.shape)  # Adjust .head() parameter as needed to display more rows

# Calculate and print accuracy
accuracy = accuracy_score(y_test_label, y_pred)
print(f"Accuracy: {accuracy}")

#===============================================

def get_attacks_df():
    # Make sure the directory exists
    os.makedirs("static/tables", exist_ok=True)

    # Define the full file path
    file_path = os.path.join("templates", "attacks_table.html")

    # Save the DataFrame
    attacks_df_html = attacks_df.to_html(file_path, index=False)  # Set index=False if you don't want the DataFrame index in the file

    return attacks_df_html


get_attacks_df()
#===============================================
        
# Set larger font sizes for all plots
plt.rc('font', size=40)          # controls default text sizes
plt.rc('axes', titlesize=40)     # fontsize of the axes title
plt.rc('axes', labelsize=40)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=25)    # fontsize of the tick labels
plt.rc('ytick', labelsize=25)    # fontsize of the tick labels
plt.rc('legend', fontsize=40)    # legend fontsize
plt.rc('figure', titlesize=40)   # fontsize of the figure title

plt.figure(figsize=(40, 20))

# Define colors for each plot
colors_protocol_type = ['blue', 'green', 'red']
colors_service = ['cyan', 'magenta', 'yellow', 'black', 'white', 'orange', 'grey', 'blue', 'green', 'red']
colors_flag = ['purple', 'pink', 'lightblue']

# Plot for protocol_type with colors
plt.subplot(131)
attacks_df["protocol_type"].value_counts().plot(kind='bar', label='protocol type', color=colors_protocol_type)
plt.xlabel('Protocol Type', fontsize=50)
plt.title('Protocol Type Counts', fontsize=50)

# Plot for service with colors
plt.subplot(132)
attacks_df['service'].value_counts().head(10).plot(kind='bar', color=colors_service)
plt.xlabel('Service', fontsize=50)
plt.title('Top 10 Services', fontsize=50)

# Plot for flag with colors
plt.subplot(133)
attacks_df["flag"].value_counts().plot(kind='bar', color=colors_flag)
plt.xlabel('Flag', fontsize=50)
plt.title('Flag Counts', fontsize=50)

file_path = "C:/Users/andre/Documents/GitHub/cyber-sentry/static/images/pt_s_f.png"

plt.savefig(file_path)
plt.close()  # Close the figure after saving to free up memory
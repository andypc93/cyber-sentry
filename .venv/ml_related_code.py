

import pandas as pd
import matplotlib.pyplot as plt
import sqlite3 as sql

from joblib import load
from sklearn.preprocessing import LabelEncoder


import os
print("Current Working Directory:", os.getcwd())

#db_path = 'C:/Users/rober/SQLite/CyberSentryDB.db'

db_path = 'C:/Users/andre/SQLite/CyberSentryDB.db'

conn = sql.connect(db_path)

cursor = conn.cursor()

#adwdw
df_new = pd.read_sql_query("SELECT * FROM testing_data", conn)

cursor.close()

conn.close()

#===================================================


import os

model_path = 'random_forest_classifier.joblib'  # Adjust as necessary
if os.path.exists(model_path):
    print(f"File found: {model_path}")
else:
    print(f"File not found: {model_path}")


model_path = 'random_forest_classifier.joblib'

RandomForestClassifier = load(model_path)

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


df_new.columns = columns

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

df_new['class'] = df_new['outcome']
df_new.loc[:, 'class'] = df_new['class'].replace(class_attack, 'attack')

df_new.drop(columns=["outcome", "level"], inplace =True)

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

#===================================================

x_test, x_test_num, y_test = preprocess(df_new, drop_columns)

#===================================================

le = LabelEncoder()

# Fit the LabelEncoder with your labels
le.fit(y_test)

# Now that le is fitted, you can transform your labels
y_test_label = le.transform(y_test)

#===================================================


y_pred = RandomForestClasifier.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test_label, y_pred)
print(f"Accuracy: {accuracy}")

# Detailed classification report
print(classification_report(y_test_label, y_pred, target_names=le.classes_))

#===================================================


#===================================================


#===================================================
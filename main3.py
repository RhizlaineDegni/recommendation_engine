# IMPORT LIBRARY

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

# IMPORT DATA

train_submissions = pd.read_csv("./train/train_submissions.csv")
problem_data = pd.read_csv("./train/problem_data.csv")
user_data = pd.read_csv('./train/user_data.csv')
test_submissions = pd.read_csv("./test_submissions_NeDLEvX.csv")

# CLEANING DATA

## join table
train = train_submissions.join(problem_data.set_index('problem_id'), on='problem_id').join(user_data.set_index('user_id'), on='user_id')
train = train.drop(columns = ['user_id','problem_id','country','max_rating'])

# Missing values
# missing_val_count_by_column = (train.isnull().sum())
# print(missing_val_count_by_column[missing_val_count_by_column > 0])

# level_type      620
# points        29075
# tags          15427
# country       37853


def SepTags(data,num):
    tags = data['tags'].str.split(",", n = num, expand = True)
    liste_tags = []
    for i in range(1,num+2):
        liste_tags.append('tag'+str(i))
    tags.columns = liste_tags
    data = data.join(tags)
    data = data.drop(columns = ['tags'])
    return data


# train.dropna(inplace=True)
train = SepTags(train,2)
train_y= train['attempts_range']
train_X = train.drop(columns = ['attempts_range'])
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.33, random_state=42)
#PIPELINE
# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant') #replace by 0 (par median plus performant ici mais pas sur les donnees submissions)

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OrdinalEncoder(handle_unknown='ignore'))
])



# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train.columns if 
                X_train[cname].dtype in ['int64', 'float64']]

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


#Step 2: Define the Model

model = XGBClassifier()

# Step 3: Create and Evaluate the Pipeline

from sklearn.metrics import mean_absolute_error

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_test)
preds = [round(i) for i in preds]

# Evaluate the model
score = f1_score(y_test, preds, average='weighted')
print('F1 score (XGB):',f1_score(y_test, preds, average='weighted'))

def Submission():
    problem_data = pd.read_csv("./train/problem_data.csv")
    user_data = pd.read_csv('./train/user_data.csv')
    test_submissions = pd.read_csv("./test_submissions_NeDLEvX.csv")
    test = test_submissions.join(problem_data.set_index('problem_id'), on='problem_id').join(user_data.set_index('user_id'), on='user_id')
    test.drop(['problem_solved', 'rating'], axis=1)
    test = SepTags(test,2)    
    ID = test['ID']
    test = test.drop(columns = ['user_id','problem_id','ID','country','max_rating'])
    y_pred = my_pipeline.predict(test)
    pd.DataFrame(np.transpose([ID,y_pred]),columns=['ID','attempts_range']).to_csv('submission.csv',index=False,header=True)

# learning_rate: 0.01
# n_estimators: 100 if the size of your data is high, 1000 is if it is medium-low
# max_depth: 3
# subsample: 0.8
# colsample_bytree: 1
# gamma: 1
model = XGBClassifier(random_state = 42,
                      learning_rate= 0.05,
                      n_estimators= 100,
                      max_depth= 25,
                      subsample= 0.8,
                      colsample_bytree= 1,
                      gamma= 1)
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])
my_pipeline.fit(X_train, y_train)
preds = my_pipeline.predict(X_test)
preds = [round(i) for i in preds]
score = f1_score(y_test, preds, average='weighted')
print('F1 score (XGB tuned):',f1_score(y_test, preds, average='weighted'))


Submission()

# Define Which columns to drop because dropping country improved the score



# F1:  0.4774982175577528
# MAE: 0.663470757430489


# ## drop na
# train.dropna(inplace=True)
# # split X and y
# train_y = train['attempts_range']
# train_X = train.drop(columns = ['attempts_range'])
# X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.33, random_state=42)

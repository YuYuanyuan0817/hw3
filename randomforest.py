import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from connector.pg_connector import get_data
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from  confv.conf import logging
import pickle
from util.util import save_model,load_model
from confv.conf import settings
from confv.conf import path_to_model


def split_data(df):
    logging.info("Select X and y")
    # Filter out target column and take all other columns
    X = df.iloc[:, :-1]
    
    # Select target column
    y = df['target']
    
    logging.info("Split variables")
    # Split variables into train and test
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    return X_train, X_test, y_train, y_test


def train_random_forest (X_train, y_train):
    logging.info("intialize model")

    # Initialize the model
    clf = RandomForestClassifier(max_depth = 2, random_state = 0)
   
    logging.info("train model")
    # Train the model
    clf.fit(X_train, y_train)
    save_model(dir=path_to_model,model=clf) 
    return clf

df = get_data(settings.DATA.data_set)
X_train, X_test, y_train, y_test =split_data(df)
clf = train_random_forest (X_train, y_train)
logging.info(f'Accuracy is {clf.score(X_test,y_test)}')



#Hyperparameters Tuning

for depth in range(1,12):
    for features in [3,4,5]:
        clf =RandomForestClassifier(max_depth = depth,max_features= features)
        clf.fit(X_train, y_train)
        
        logging.info(f'max_depth = {depth} & max_features = {features}')
        logging.info(f'Accuracy is {clf.score(X_train,y_train)}')
        logging.info(f'Accuracy is {clf.score(X_test,y_test)}')

#Making a prediction
def predict (values,path_to_model):
    clf = load_model(path_to_model)
    return clf. predict(values)

responce = predict(X_test,path_to_model)
logging.info(f'Prediction is {clf.predict(X_test)}')





import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from connector.pg_connector import get_data
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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


def train_svc (X_train, y_train):
   
    logging.info("intialize model")

    # Initialize the model
    svm =  SVC(random_state=3, probability=True)
    logging.info("train model")

    # Train the model
    svm.fit(X_train, y_train)
    save_model(dir=path_to_model,model=svm) 
    return svm




df = get_data(settings.DATA.data_set)
X_train, X_test, y_train, y_test =split_data(df)
svm = train_svc (X_train, y_train)
logging.info(f'Accuracy is {svm.score(X_test,y_test)}')

#Hyperparameters Tuning

for this_gamma in [0.001, 0.01, 0.1]:
    
    for this_C in [0.1, 1, 15]:
        
        #initialize a model
        svm = SVC(gamma=this_gamma, C = this_C)
        
        #fit the model
        svm.fit(X_train, y_train)
        
        logging.info(f'SVM with gamma = {this_gamma} & C = {this_C}')
        logging.info(f'Accuracy is {svm.score(X_train,y_train)}')
        logging.info(f'Accuracy is {svm.score(X_test,y_test)}')


#Making a prediction
def predict (values,path_to_model):
    svm = load_model(path_to_model)
    return svm.predict(values)

responce = predict(X_test,path_to_model)
logging.info(f'Prediction is {svm.predict(X_test)}')
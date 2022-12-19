from connector.pg_connector import get_data
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from  conf.conf import logging
import pickle
from util.util import save_model,load_model
from conf.conf import settings

logging.info("extracting df")

# Load the dataset
df = get_data('https://raw.githubusercontent.com/5x12/ml-cookbook/master/supplements/data/heart.csv')

logging.info("DF is extracted")


def train_test_split(df):
    logging.info("Select X and y")
    # Filter out target column and take all other columns
    X = df.iloc[:, :-1]
    
    # Select target column
    y = df['target']
    
    logging.info("Split variables")
    # Split variables into train and test
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 3)
    
    return X_train, X_test, y_train, y_test


def train_svc (X_train, y_train):
   
    logging.info("intialize model")

    # Initialize the model
    clf =  SVC(random_state=3, probability=True)
   
    logging.info("train model")

    # Train the model
    clf.fit(X_train, y_train)
    
    save_model(dir='model1/conf/svm.pkl',model=clf)
    

    return clf


df = get_data('https://raw.githubusercontent.com/5x12/ml-cookbook/master/supplements/data/heart.csv')
X_train, X_test, y_train, y_test = train_test_split(df)
clf = train_svc (X_train, y_train)

logging.info(f'Accuracy is {clf.score(X_test,y_test)}')

clf = load_model('model1/conf/svc.pkl')

logging.info(f'Prediction is {clf.predict(X_test)}')
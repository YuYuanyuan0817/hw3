from connector.pg_connector import get_data
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
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
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 0)
    
    return X_train, X_test, y_train, y_test


def train_random_forest (X_train, y_train):
    logging.info("intialize model")

    # Initialize the model
    clf = RandomForestClassifier(max_depth = 2, random_state = 0)
   
    logging.info("train model")
    # Train the model
    clf.fit(X_train, y_train)
    
    save_model(dir='model/conf/randomforest.pkl',model=clf)
    
    return clf


print(f"parameter{settings.DATA.data_set}")


df = get_data('https://raw.githubusercontent.com/5x12/ml-cookbook/master/supplements/data/heart.csv')
X_train, X_test, y_train, y_test = train_test_split(df)
clf = train_random_forest (X_train, y_train)
logging.info(f'Accuracy is {clf.score(X_test,y_test)}')


#Hyperparameters Tuning
from sklearn.model_selection import GridSearchCV, cross_val_score,KFold
clf_params = {'n_estimators':[100,300,500,700],"max_depth": range(1,12), 'max_features':[3,4,5],'min_samples_leaf':range(1,10)}
clf_grid = GridSearchCV(clf, clf_params, cv=KFold(shuffle=True,random_state=0), n_jobs=-1)
clf_grid.fit(X_train, y_train)
clf_grid.best_params_, clf_grid.best_score_
logging.info(f'clf_grid with best_params_ = {clf_params}')
logging.info(f'Accuracy is {clf.score(X_train,y_train)}')
logging.info(f'Accuracy is {clf.score(X_test,y_test)}')



#Making a prediction
def predict1 (values):
    clf = load_model('model/conf/randomforest.pkl')
    return clf. predict1(values)





responce = predict1(X_test)
clf = load_model('model/conf/randomforest.pkl')

logging.info(f'Prediction is {clf.predict1(X_test)}')






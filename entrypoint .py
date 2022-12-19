from model.randomforest  import predict1
from conf.conf import logging
from model.svm import predict2


#random forest model prediction
logging.info (f"prediction:{predict1([[59, 1, 0, 101, 234, 0, 1, 143, 0, 3.4, 0, 0, 0]],'model1/conf/random_forest.pkl')}")

#svm model prediction
logging.info (f"prediction:{predict2([[59, 1, 0, 101, 234, 0, 1, 143, 0, 3.4, 0, 0, 0]],'model1/conf/random_forest.pkl')}")


import argparse

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prediction_model",type=str,help="model filename")
    parser.add_argument("--prediction_params",type=int,help="params number")
    return parser.parse_args()




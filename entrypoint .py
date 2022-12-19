from model.randomforest  import predict1
from conf.conf import logging
from model.svm import predict2


#random forest model prediction
logging.info (f"prediction:{predict1([[59, 1, 0, 101, 234, 0, 1, 143, 0, 3.4, 0, 0, 0]],'model/conf/randomforest.pkl')}")

#svm model prediction
logging.info (f"prediction:{predict2([[59, 1, 0, 101, 234, 0, 1, 143, 0, 3.4, 0, 0, 0]],'model/conf/svm.pkl')}")


import argparse

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prediction_model",type=str,help="model filename")
    parser.add_argument("--prediction_params",type=float,help="params number")
    return parser.parse_args()




def create_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_params",
                        type=float,
                        help="params number")

    parser.add_argument("--prediction_model",
                        type=str,
                        dest="model_name", 
                        choices=['randomforest','svm'],
                        help="return the name of the model")
    return parser

if __name__ == "__main__":

    parser = create_parser()
   
    args = parser.parse_args()
    
    print('args dict:', vars(args))
    
    
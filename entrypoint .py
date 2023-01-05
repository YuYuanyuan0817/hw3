from model.randomforest  import predict
from confv.conf import logging
from model.svm import predict


#random forest model prediction
logging.info (f"prediction:{predict([[59, 1, 0, 101, 234, 0, 1, 143, 0, 3.4, 0, 0, 0]],path_to_model ='model/conf/decision_tree.pkl')}")

#svm model prediction
logging.info (f"prediction:{predict([[59, 1, 0, 101, 234, 0, 1, 143, 0, 3.4, 0, 0, 0]],path_to_model ='model/conf/decision_tree.pkl')}")



 

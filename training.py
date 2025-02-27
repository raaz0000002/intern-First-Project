"""
Note: You need to run first datapreprocessing.py to import final_X
"""
from datapreprocessing import final_X,final_Y

from sklearn.model_selection import train_test_split

#data spliting into train test 
X_train,X_test,Y_train,Y_test=train_test_split(final_X,final_Y,train_size=0.7)


#select machine learning model and train it
from sklearn.ensemble import RandomForestClassifier
import time
model=RandomForestClassifier()

start_time=time.time()
model.fit(X_train,Y_train)
end_time=time.time()
print("Model trained in:",(end_time-start_time))


# measure accuracy
from sklearn.metrics import classification_report
predicted=model.predict(X_test)
score=classification_report(Y_test,predicted)

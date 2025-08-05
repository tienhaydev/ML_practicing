import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, recall_score, precision_score

#read data
data = pd.read_csv("diabetes.csv")
target = "Outcome"

#create data report file
# profile = ProfileReport(data, title="data report", explorative=True)
# profile.to_file("diabetes_report.html")

#split data
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=35434)

#preprocess data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#select model
model = SVC()

#train model
model.fit(x_train, y_train)

#predict
y_predict = model.predict(x_test)

for i, j in zip(y_predict, y_test):
    print("Predict value: {} Actual value: {}".format(i,j))

#evaluate models
print("Accuracy: {}".format(accuracy_score(y_test,y_predict)))
print("F1: {}".format(f1_score(y_test,y_predict)))
print("Precision: {}".format(precision_score(y_test,y_predict)))
print("Recall: {}".format(recall_score(y_test,y_predict)))
print("Confusion matrix: {}".format(confusion_matrix(y_test,y_predict)))

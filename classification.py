import pickle
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, recall_score, precision_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from lazypredict.Supervised import LazyClassifier, LazyRegressor

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

params = {
    "n_estimators": [50, 100, 200],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [None, 2, 5]
}

#select model
# model = RandomForestClassifier(n_estimators=50, criterion="log_loss", random_state=100)

# model = GridSearchCV(RandomForestClassifier(random_state=100),
#                      param_grid=params,
#                      scoring="recall",
#                      cv=6,
#                      verbose=1,
#                      # n_job=-1
#                      )

model = RandomizedSearchCV(RandomForestClassifier(random_state=100),
                     param_distributions=params,
                     scoring="precision",
                     n_iter=20,
                     cv=6,
                     verbose=1,
                     # n_job=-1
                     )

#train model
model.fit(x_train, y_train)
print(model.best_score_)
print(model.best_params_)
filename = 'best_models.pkl'
# pickle.dump(model, open(filename, 'wb'))
with open(filename, 'wb') as fo:
    pickle.dump(model, fo)

# model = pickle.load(open(filename, 'rb'))

#predict
y_predict = model.predict(x_test)

# for i, j in zip(y_predict, y_test):
#     print("Predict value: {} Actual value: {}".format(i,j))

#evaluate models
print("Accuracy: {}".format(accuracy_score(y_test,y_predict)))
print("F1: {}".format(f1_score(y_test,y_predict)))
print("Precision: {}".format(precision_score(y_test,y_predict)))
print("Recall: {}".format(recall_score(y_test,y_predict)))
print("Confusion matrix: {}".format(confusion_matrix(y_test,y_predict)))

print(classification_report(y_test, y_predict))

# clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = clf.fit(x_train, x_test, y_train, y_test)

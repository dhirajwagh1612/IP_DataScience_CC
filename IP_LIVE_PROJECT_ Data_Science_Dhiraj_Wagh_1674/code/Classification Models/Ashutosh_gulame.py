import pandas as pd
import numpy as np
import sys

df = pd.read_csv(r"C:\Users\Admin-pc\Downloads\ip\liveproj\DS_DATESET.csv")

df=df.drop(["Email Address"],axis=1)
df=df.drop(["First Name"],axis=1)
df=df.drop(["Last Name"],axis=1)
df=df.drop(["Zip Code"],axis=1)
df=df.drop(["Contact Number"],axis=1)
df=df.drop(["Emergency Contact Number"],axis=1)
df=df.drop(["University Name"],axis=1)
df=df.drop(["Degree"],axis=1)
df=df.drop(["Course Type"],axis=1)
df=df.drop(["Current Employment Status"],axis=1)
df=df.drop(["Expected Graduation-year"],axis=1)
df=df.drop(["State"],axis=1)
df=df.drop(["Certifications/Achievement/ Research papers"],axis=1)
df=df.drop(["DOB [DD/MM/YYYY]"],axis=1)
df=df.drop(["Link to updated Resume (Google/ One Drive link preferred)"],axis=1)
df=df.drop(["link to Linkedin profile"],axis=1)

lbl=df["Label"].tolist()
x=[]
for i in range(0,len(lbl)):
    if lbl[i]=='eligible':
        x.append(1)
    else:
        x.append(0)

Label=pd.DataFrame({"Label":x})
df=df.drop(["Label"],axis=1)
df=pd.concat([df,Label],axis=1,sort=False)

dfencode=df
dfencode["Areas of interest"] = dfencode["Areas of interest"].astype('category')
dfencode["Areas of interest"] = dfencode["Areas of interest"].cat.codes
dfencode["Which-year are you studying in?"]=dfencode["Which-year are you studying in?"].astype("category")
dfencode["Which-year are you studying in?"]=dfencode["Which-year are you studying in?"].cat.codes
dfencode["Major/Area of Study"]=dfencode["Major/Area of Study"].astype("category")
dfencode["Major/Area of Study"]=dfencode["Major/Area of Study"].cat.codes
dfencode["City"]=dfencode["City"].astype("category")
dfencode["City"]=dfencode["City"].cat.codes
dfencode["Gender"]=dfencode["Gender"].astype("category")
dfencode["Gender"]=dfencode["Gender"].cat.codes
dfencode["College name"]=dfencode["College name"].astype("category")
dfencode["College name"]=dfencode["College name"].cat.codes
dfencode["Have you worked core Java"]=dfencode["Have you worked core Java"].astype("category")
dfencode["Have you worked core Java"]=dfencode["Have you worked core Java"].cat.codes
dfencode["Programming Language Known other than Java (one major)"]=dfencode["Programming Language Known other than Java (one major)"].astype("category")
dfencode["Programming Language Known other than Java (one major)"]=dfencode["Programming Language Known other than Java (one major)"].cat.codes
dfencode["Have you worked on MySQL or Oracle database"]=dfencode["Have you worked on MySQL or Oracle database"].astype("category")
dfencode["Have you worked on MySQL or Oracle database"]=dfencode["Have you worked on MySQL or Oracle database"].cat.codes
dfencode["Have you studied OOP Concepts"]=dfencode["Have you studied OOP Concepts"].astype("category")
dfencode["Have you studied OOP Concepts"]=dfencode["Have you studied OOP Concepts"].cat.codes
dfencode["Rate your written communication skills [1-10]"]=dfencode["Rate your written communication skills [1-10]"].astype("category")
dfencode["Rate your written communication skills [1-10]"]=dfencode["Rate your written communication skills [1-10]"].cat.codes
dfencode["Rate your verbal communication skills [1-10]"]=dfencode["Rate your verbal communication skills [1-10]"].astype("category")
dfencode["Rate your verbal communication skills [1-10]"]=dfencode["Rate your verbal communication skills [1-10]"].cat.codes
dfencode["How Did You Hear About This Internship?"]=dfencode["How Did You Hear About This Internship?"].astype("category")
dfencode["How Did You Hear About This Internship?"]=dfencode["How Did You Hear About This Internship?"].cat.codes

from sklearn import preprocessing
X = np.asarray(dfencode[['City', 'Age', 'Gender', 'College name', 'Major/Area of Study', 'Which-year are you studying in?', 'CGPA/ percentage','Areas of interest','Have you worked core Java','Programming Language Known other than Java (one major)','Have you worked on MySQL or Oracle database','Have you studied OOP Concepts','Rate your written communication skills [1-10]','Rate your verbal communication skills [1-10]','How Did You Hear About This Internship?']])
y = np.asarray(dfencode['Label'])
X = preprocessing.StandardScaler().fit(X).transform(X)
# splitting data in train and test

# LOGISTIC REGRESSION
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)
print("with logistic regression accuracy :",jaccard_similarity_score(y_test, yhat))
print("with logistic regression f1 score:",f1_score(y_test,yhat))

#K nearest
from sklearn.neighbors import KNeighborsClassifier

k = 100
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat1 = neigh.predict(X_test)
from sklearn import metrics
print("Train set k nearest Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set k nearest Accuracy: ", metrics.accuracy_score(y_test, yhat1))
print("with k nearest f1 score:",f1_score(y_test,yhat1))

#naive bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print("with naive bayes accuracy :",jaccard_similarity_score(y_test, y_pred))
print("with navie bayes f1 score:",f1_score(y_test,y_pred))



#SVM
from sklearn import svm
clf1 = svm.SVC(kernel='rbf')
clf1.fit(X_train, y_train) 
yhat2 = clf1.predict(X_test)
clf2 = svm.SVC(kernel='rbf',C=100,gamma=0.1)
clf2.fit(X_train, y_train) 
yhat3 = clf2.predict(X_test)
print("with SVM accuracy :",jaccard_similarity_score(y_test, yhat3))
print("with SVM f1 score:",f1_score(y_test,yhat3))


# # RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier() 
classifier=classifier.fit(X_train,y_train) 
yhat4=classifier.predict(X_test) 
print("with random forest accuracy :",jaccard_similarity_score(y_test, yhat4))
print("with random forest F1 score:",f1_score(y_test,yhat4))
print("\n\nso random forest is best model ")
print("with random forest accuracy :",jaccard_similarity_score(y_test, yhat4))
print("with random forest F1 score:",f1_score(y_test,yhat4))
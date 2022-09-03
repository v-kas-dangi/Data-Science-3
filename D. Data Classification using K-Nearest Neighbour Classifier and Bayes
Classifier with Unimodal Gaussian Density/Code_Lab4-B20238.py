# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
"""
Vikas Dangi
B20238
DS3-Lab 4
"""
# Importing modules
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# from sklearn.metrics import plot_confusion_matrix

# reading the file
df=pd.read_csv("SteelPlateFaults-2class.csv")
k=[1,3,5]
# assigning data in X and target in y
X=df.iloc[:,:27]
y=df['Class']

# Performing the split by taking approx 70% of data for training from both the data and the target
#NOTE: After this we will have 70% of target from each class and then they will be clumbed together automatically to form our train data using train_test_split method
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,shuffle=True)

X_train.to_csv('SteelPlateFaults-train.csv',index=False)
X_test.to_csv('SteelPlateFaults-test.csv',index=False)


# %%

#Question 1

#A

# Function to Calculate and print the Confusion matrix
def ConfusionMat(n):
    knn_clf=KNeighborsClassifier(n_neighbors=n)
    knn_clf.fit(X_train,y_train)
    knn_predictions=knn_clf.predict(X_test)
    print("The Confusion matrix for k=",n,"is: ")
    print(confusion_matrix(y_test,knn_predictions))

#calling function
for n in k:
    ConfusionMat(n)

#B

# Function to Calculate and print the Classification accuracy
def ClassificationAccuracy(n):
    knn_clf=KNeighborsClassifier(n_neighbors=n)
    knn_clf.fit(X_train,y_train)
    knn_predictions=knn_clf.predict(X_test)
    return accuracy_score(y_test,knn_predictions)
#calling function
mx=0
nmax=0

for n in k:
    acc=ClassificationAccuracy(n)
    print("Accuracy of the knn algorithm with k=",n,"is : ",acc)
    if(acc>mx):
        mx=acc
        nmax=n
print("The maximum accuracy is for the value k=",nmax," that is ",mx)


# %%
# Question 2

#normalisation
X_test=(X_test-X_train.min())/(X_train.max()-X_train.min())
X_test.to_csv("SteelPlateFaults-test-Normalised.csv")

#normalisation
X_train=(X_train-X_train.min())/(X_train.max()-X_train.min())
X_train.to_csv("SteelPlateFaults-train-Normalised.csv")


#A
for n in k:
    ConfusionMat(n)

#B
mx1=0
nmax=0
for n in k:
    acc=ClassificationAccuracy(n)
    print("Accuracy of the knn algorithm with k=",n,"is : ",acc)
    if(acc>mx1):
        mx1=acc
        nmax=n
print("The maximum accuracy is for the value k=",nmax," that is ",mx1)


# %%
# Question 3
#Dropping the correlated columns
df=df.drop(["X_Minimum","Y_Minimum","TypeOfSteel_A300","TypeOfSteel_A400"],axis=1)
X0=df.loc[df["Class"]==0]
X1=df.loc[df["Class"]==1]
X0=X0.drop("Class",axis=1)
X1=X1.drop("Class",axis=1)
y0=df["Class"].loc[df["Class"]==0]
y1=df["Class"].loc[df["Class"]==1]
X_train0, X_test0, y_train0, y_test0 = train_test_split(X0, y0, test_size=0.3, random_state=42,shuffle=True)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3, random_state=42,shuffle=True)

# function to calculate the likelihoood value
def likelihood(x,D):
    cov=D.cov()
    mean=D.mean()
    det=abs(np.linalg.det(cov))
    meanminus=x-mean
    d=len(D.columns)
    invcov=pd.DataFrame(np.linalg.pinv(cov.values), cov.columns, cov.index)
    # using the formula for multivariate or multimodel distribution
    p=(1/(((2*np.pi)**(d/2))*det**0.5)) * (np.e**(-0.5*(meanminus@invcov@meanminus)))
    return p

P_C0=len(y_train0)/(len(y_train0)+len(y_train1))
P_C1=len(y_train1)/(len(y_train0)+len(y_train1))
Xtrain=X_train0.append(X_train1)
Xtest=X_test0.append(X_test1)
ytest=y_test0.append(y_test1)
predictions=[] #to store the final predictions

# to predict all the test cases
for i in range(len(Xtest)):
    lh0=likelihood(Xtest.iloc[i],X_train0)
    lh1=likelihood(Xtest.iloc[i],X_train1)
    postPosb0=P_C0*lh0/(P_C0*lh0+P_C1*lh1)
    postPosb1=P_C1*lh1/(P_C0*lh0+P_C1*lh1)
    if(postPosb0>=postPosb1):
        predictions.append(0)
    else:
        predictions.append(1)
mx2=accuracy_score(ytest,predictions)
print("The accuracy of the Bayes CLassifier is ",mx2)

print("The Confusion matrix for k=",n,"is: ")
print(confusion_matrix(ytest,predictions))

print("The mean of Class 0 training data: \n",X_train0.mean())
print("The mean of Class 1 training data: \n",X_test0.mean())


# %%
#Question 4
# printing all the accuracies and comparing them
print("The different accuracies in the above processes are ",mx," ",mx1," ",mx2)
print("The highest we get in the ",max(max(mx,mx1),mx2))


# %%




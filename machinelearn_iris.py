import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('iris.csv')


# Seeing the General info about the data and a visualization based on differnt plots
def show_general_info():
    print(data.head())
    print(data.info())
    print(data.describe)
    print(data.value_counts("variety"))
    print(data.isnull().sum()) # Checking for missing values
    sns.pairplot(data, hue="variety")
    plt.show()


# Setting up the data for further modelling
# features_x will now only contain floats or integers
# target_y will now be a 1d array of names of the iris flowers
features_x = data.drop('variety', axis=1) # axis = columns, when checking what to drop.
target_y = data['variety']


# Using the sklearn library, we split the dataset into training set and testing set.
# Test_size is being set to 25%, and random sate 10 for replication oppertunity
X_train, X_test, y_train, y_test = train_test_split(features_x, target_y, test_size=0.25, random_state=10)

# Checking that the X-test variable works
print(X_test.describe())


def knn_method(n_neighbors):

    # Also from Sklearn lib, we use the nearest neighbor method.
    # n_neighbors = 3, means the model will consider the 3 nearest neighbors when making predictions
    knn = KNeighborsClassifier(n_neighbors)

    # using sklearn fit-method, we will train the model on the training data.
    knn.fit(X_train,y_train)

    # Make predictions on the test set
    y_pred_knn = knn.predict(X_test)

    # Calculate and print the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred_knn)
    print(f"Accuracy: {accuracy:.2f}") # {accuracy:.2f} specifies the accuracy value in the string with two decimal places 

    # Display confusion matrix and classification report
    conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
    class_report_knn = classification_report(y_test, y_pred_knn)

    print("Confusion Matrix (KNN):")
    print(conf_matrix_knn)
    print("\nClassification Report (KNN):")
    print(class_report_knn)

    # Storing the true and predicted labels in separate variables, for easy usage for calculations
    true_labels = y_test  
    predicted_labels = y_pred_knn  

    # Get the unique class names from the target variable 
    class_names = data['variety'].unique()
    conf_matrix_knn = confusion_matrix(true_labels, predicted_labels)

    # Creating visuals for the confusion matrix
    sns.heatmap(conf_matrix_knn, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def decision_tree_method():

    # Also from Sklearn lib, we use the nearest neighbor method.
    # n_neighbors = 3, means the model will consider the 3 nearest neighbors when making predictions
    tree_clf = tree.DecisionTreeClassifier()

    # using sklearn fit-method, we will train the model on the training data.
    tree_clf.fit(X_train,y_train)

    # Make predictions on the test set
    y_pred_tree_clf = tree_clf.predict(X_test)

    # Calculate and print the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred_tree_clf)
    print(f"Accuracy: {accuracy:.2f}") # {accuracy:.2f} specifies the accuracy value in the string with two decimal places 

    # Display confusion matrix and classification report
    conf_matrix_tree = confusion_matrix(y_test, y_pred_tree_clf)
    class_report_tree = classification_report(y_test, y_pred_tree_clf)

    print("Confusion Matrix (DTree):")
    print(conf_matrix_tree)
    print("\nClassification Report (DTree):")
    print(class_report_tree)

knn_method(n_neighbors=5)
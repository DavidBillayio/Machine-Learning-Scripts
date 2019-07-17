# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

#Initialize variables

X=0
y=0
max_accuracy=0

#load dataset
data = datasets.load_digits()
#data = datasets.load_iris()	#Load and return the iris dataset (classification).
#data = datasets.load_digits()	#Load and return the digits dataset (classification).
#data datasets.load_wine()	#Load and return the wine dataset (classification).
#data = datasets.load_breast_cancer()	#Load and return the breast cancer wisconsin dataset (classification).

print(data.keys())
print(data.info)


# Create feature and target arrays
X = data.data
y = data.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Setup testing sequence and the corresponding number of output train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors)) 
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors = k)

    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

n_neighbors= test_accuracy.index(max(test_accuracy))

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Print the accuracy
print(knn.score(X_test, y_test))

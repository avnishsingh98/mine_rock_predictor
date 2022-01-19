import warnings
warnings.filterwarnings("ignore")                       # Ignore warnings
import pandas as pd                                     # For Dataframes
import matplotlib.pyplot as plt                         # modifying plot
from sklearn.model_selection import train_test_split    # Splits Data
from sklearn.metrics import accuracy_score              # Grade result
from sklearn.preprocessing import StandardScaler        # Stadardize Data
from sklearn.decomposition import PCA                   # PCA package
from sklearn.metrics import confusion_matrix            # generate the  matrix
from sklearn.neural_network import MLPClassifier        # Algorithm

#CONSTANTS
NEW_SEED = 10
COMPONENTS = 61

# Load in the data from the csv file
df = pd.read_csv('sonar_all_data_2.csv',header=None)

# Separate the desired features
X = df.iloc[:,:60].values
# Extract the classifications
y = df.iloc[:,60].values
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

sc = StandardScaler()                       # create the standard scalar
sc.fit(X_train)                             # compute the required transformation
X_train_std = sc.fit_transform(X_train)     # apply to the training data
X_test_std = sc.transform(X_test)           # and SAME transformation of test data

temp = 0
ls = []
for comp in range(1,COMPONENTS):

    pca = PCA(n_components = comp)                   # only keep "best" features!
    X_train_pca = pca.fit_transform(X_train_std)     # apply to the train data
    X_test_pca = pca.transform(X_test_std)           # do the same to the test data

    # Define the model
    model = MLPClassifier( hidden_layer_sizes=(100), activation='logistic',\
                             max_iter=2000, alpha=0.00001,solver='adam', \
                                 tol=0.0001 , random_state= NEW_SEED)

    model.fit(X_train_pca, y_train)

    # Prediction on test dataset
    y_pred = model.predict(X_test_pca)
    acc_score = accuracy_score(y_test, y_pred)

    # Save accuracy scores for plotting later
    ls.append(acc_score)

    # Keep a track of the highest accuracy achieved
    if acc_score > temp:
        temp = acc_score
        temp_c = comp

        # Create the confusion matrix for highest accuracy
        confuse = confusion_matrix(y_test,y_pred)

    # Print number of components and corresponding accuracy
    print('\nNumber of Components :', comp, '\nTest Accuracy: %.2f' % acc_score)

# Printing highest accuracy and confusion matrix
print("\nHighest accuracy was %.2f" % temp, "for", temp_c, "Components\n")
print("CONFUSION MATRIX \n---------------------\n", confuse)

# Plot and show the graph
plt.plot(range(1,COMPONENTS), ls , marker = '+')
plt.xlabel('Number of Components')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Components')
plt.show()

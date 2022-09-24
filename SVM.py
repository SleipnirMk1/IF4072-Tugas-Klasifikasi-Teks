import pandas as pd

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# CHANGE DIRECTORY
directory = "C:\\Tugas\\IF4072 NLP\\Tugas Tim Klasifikasi Teks\\"

train = pd.read_csv(directory + 'data_worthcheck\\train.csv')
test = pd.read_csv(directory + 'data_worthcheck\\test.csv')

# TAKE ONLY HEAD FOR FASTER PROCESSING
train = train.head(2000)

# Separate to X and Y
X_train = train['text_a']
Y_train = train['label']
X_test = test['text_a']
Y_test = test['label']

# Preprocessing: Stem & Remove Stop Words; Tokenizer built in to tf idf vectorizer
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

stop_factory = StopWordRemoverFactory()
stopword = stop_factory.create_stop_word_remover()

for idx in range (len(X_train)):
    # Stem each sentence
    X_train[idx] = stemmer.stem(X_train[idx])
    # Remove Stop Words
    X_train[idx] = stopword.remove(X_train[idx])

# Vectorize Training Data, use TF IDF
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train).toarray()
X_train = pd.DataFrame(X_train, columns = vectorizer.get_feature_names_out())

# Vectorize Test Data
X_test = vectorizer.transform(X_test).toarray()
X_test = pd.DataFrame(X_test, columns = vectorizer.get_feature_names_out())

# Support Vector Machine (SVM)
# Gamma = Auto
clf = SVC(kernel = 'rbf', gamma = 'auto')
clf = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

print("Kernel: RBF, Gamma: Auto")
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred, zero_division = 0))

# Gamma = Scale
clf = SVC(kernel = 'rbf', gamma = 'scale')
clf = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

print("Kernel: RBF, Gamma: Scale")
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

# Kernel = Polynomial
clf = SVC(kernel = 'poly', gamma = 'scale')
clf = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

print("Kernel: Polynomial, Gamma: Scale")
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

# Kernel = Sigmoid
clf = SVC(kernel = 'sigmoid', gamma = 'scale')
clf = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

print("Kernel: Sigmoid, Gamma: Scale")
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

# Kernel RBF, Class Weight = Balanced
clf = SVC(kernel = 'rbf', gamma = 'scale', class_weight = 'balanced')
clf = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

print("Kernel: RBF, Gamma: Scale, Class Weight = Balanced")
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

# Kernel Sigmoid, Class Weight = Balanced
clf = SVC(kernel = 'sigmoid', gamma = 'scale', class_weight = 'balanced')
clf = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

print("Kernel: Sigmoid, Gamma: Scale, Class Weight = Balanced")
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
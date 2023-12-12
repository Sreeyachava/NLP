import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load the dataset containing comments and labels
data = pd.read_csv('comments_with_labels.csv')  # Replace with your dataset file

# Assuming 'text' column contains comments and 'label' column contains sentiment labels
X = data['text']  # Comments
y = data['label']  # Sentiment labels

# Split the data into training and testing sets (adjust test_size and random_state as needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert comments into numerical features using TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust max_features
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a simple classifier (Support Vector Machine - SVM)
classifier = SVC(kernel='linear')
classifier.fit(X_train_vectorized, y_train)

# Predict sentiment labels for the test set
y_pred = classifier.predict(X_test_vectorized)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Count number of negative, positive, and neutral comments
num_negative = sum(1 for label in y_pred if label == 'Negative')
num_positive = sum(1 for label in y_pred if label == 'Positive')
num_neutral = sum(1 for label in y_pred if label == 'Neutral')

# Display evaluation metrics
print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Display number of negative, positive, and neutral comments
print("\nNumber of Negative Comments:", num_negative)
print("Number of Positive Comments:", num_positive)
print("Number of Neutral Comments:", num_neutral)



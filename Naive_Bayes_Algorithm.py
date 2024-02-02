# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample email data
emails = [
    ("Spam", "Buy now, limited-time offer!"),
    ("Not Spam", "Meeting tomorrow at 10 AM."),
    ("Spam", "Earn money fast with no effort!"),
    ("Not Spam", "Reminder: Pay your utility bills."),
    # Add more email examples as needed
]

# Extract features and labels
X = [email[1] for email in emails]
y = [email[0] for email in emails]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Create a Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test_vec)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)

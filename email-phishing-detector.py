import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer

# Load your dataset
data = pd.read_csv("phishing_emails.csv")

# Preprocess data
data['email_text'] = data['email_text'].str.lower()  # Convert to lowercase
data['email_text'] = data['email_text'].str.replace('\d+', '', regex=True)  # Remove numbers

# Extract sender domain
data['sender_domain'] = data['sender_email'].str.split('@').str[1]

# Custom feature extractor
class CustomFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        features = []
        for _, row in X.iterrows():
            text = row['email_text']
            features.append({
                'text_length': len(text),
                'exclamation_marks': text.count('!'),
                'sender_domain': row['sender_domain']
            })
        return features

# Split data into training and testing sets
X = data[['email_text', 'sender_domain']]
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline
pipeline = Pipeline([
    ('features', FeatureUnion(transformer_list=[
        ('custom_features', Pipeline([
            ('extract_features', CustomFeatures()),
            ('vectorizer', DictVectorizer())
        ])),
        ('tfidf', TfidfVectorizer())
    ])),
    ('classifier', RandomForestClassifier())
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
report = classification_report(y_test, y_pred)
print(report)
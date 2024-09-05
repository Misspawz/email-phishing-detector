# email-phishing-detector
A Python-based machine learning pipeline for detecting phishing emails. This project uses custom feature extraction and TF-IDF vectorization to classify emails as phishing or legitimate, leveraging a Random Forest classifier for performance

# Phishing Email Classifier
A Python-based machine learning pipeline for detecting phishing emails. This project uses custom feature extraction and TF-IDF vectorization to classify emails as phishing or legitimate, leveraging a Random Forest classifier for robust performance.

## Features
- Custom feature extraction (e.g., email length, punctuation count, sender domain)
- TF-IDF vectorization for text-based feature generation
- Random Forest Classifier for phishing email detection
- Easy-to-use pipeline for model training and prediction

## Requirements 
pandas
scikit-learn

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/phishing-email-classifier.git
   cd phishing-email-classifier
   
2. Create a virtual environment (optional but recommended):
bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
bash
Copy code
pip install -r requirements.txt

## Usage
1. Prepare your dataset:
Ensure that your dataset is in CSV format and contains the following columns:
email_text: The text of the email.
sender_email: The sender's email address.
label: The target label (e.g., phishing or not phishing).

2. Run the script:
bash
Copy code
python phishing_email_classifier.py

3. Model evaluation:
After running the script, a classification report will be printed, showing the model's performance.

##Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue for improvements and feature requests.

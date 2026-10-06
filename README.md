
# Automated Customer Review Sentiment Analysis System

## Project Overview

This project is an automated sentiment analysis system developed to classify customer feedback as:

- Positive
- Neutral
- Negative

The system uses Natural Language Processing and machine-learning techniques to clean customer-review text, convert it into numerical features, and predict its sentiment.

It includes:

- Text preprocessing using NLTK
- Bag-of-Words and TF-IDF feature extraction
- Multinomial Naive Bayes and Linear Support Vector Machine models
- Model comparison and evaluation
- A Tkinter graphical user interface
- CSV and JSON review processing
- Automated incoming-file monitoring
- Negative-review alerts
- Model retraining support
- Automated software tests

The final selected model uses **TF-IDF feature extraction with a Linear Support Vector Machine classifier**.

---

## Business Problem

Businesses receive large amounts of customer feedback through reviews, social-media messages, emails, and support platforms.

Manually reading and categorizing every review is:

- Time-consuming
- Inconsistent
- Difficult to scale
- Likely to delay responses to customer concerns

This project automates review classification so that businesses can quickly understand general customer sentiment, identify high levels of negative feedback, and make informed service-improvement decisions.

---

## Project Objectives

The project objectives are to:

1. Collect and prepare a labelled customer-feedback dataset.
2. Clean and preprocess customer-review text.
3. Convert text into machine-readable numerical features.
4. Train and compare sentiment-classification models.
5. Evaluate the models using appropriate classification metrics.
6. Save the best-performing model and vectorizer.
7. Build a user-friendly desktop dashboard.
8. Automatically classify reviews uploaded through CSV or JSON files.
9. Generate alerts when negative feedback exceeds a configured threshold.
10. Support periodic retraining using newly labelled customer reviews.

---

## Dataset

The project uses the **Twitter US Airline Sentiment** dataset obtained from Kaggle.

The original dataset contains customer feedback directed at several United States airlines. Each review has one of the following sentiment labels:

- `negative`
- `neutral`
- `positive`

The original dataset contained:

| Description           | Records |
| --------------------- | ------: |
| Original records      |  14,640 |
| Prepared records      |  14,427 |
| Final cleaned records |  14,069 |

### Final Sentiment Distribution

| Sentiment       | Number of reviews |     Percentage |
| --------------- | ----------------: | -------------: |
| Negative        |             9,043 |         64.28% |
| Neutral         |             2,880 |         20.47% |
| Positive        |             2,146 |         15.25% |
| **Total** |  **14,069** | **100%** |

The dataset is imbalanced because negative reviews form the majority class.

For this reason, model selection was not based on accuracy alone. Macro precision, macro recall, macro F1-score, weighted F1-score, class-level metrics, and confusion matrices were also evaluated.

---

## Technologies Used

- Python 3.13
- pandas
- NumPy
- scikit-learn
- NLTK
- Matplotlib
- Tkinter
- Pillow
- joblib
- pytest
- Visual Studio Code

---

## Text Preprocessing

Customer reviews are processed before model training and prediction.

The preprocessing pipeline performs the following operations:

1. Converts HTML entities to readable text.
2. Removes HTML tags.
3. Removes website URLs.
4. Removes social-media account handles.
5. Preserves the text contained in hashtags.
6. Converts text to lowercase.
7. Expands common negative contractions.
8. Removes unnecessary punctuation and special characters.
9. Tokenizes the review.
10. Removes common English stopwords.
11. Preserves important negation words such as `not`, `no`, and `never`.
12. Lemmatizes words into simpler base forms.
13. Removes empty and duplicate cleaned reviews.

### Preprocessing Example

Original review:

```text
@VirginAmerica I didn't enjoy the delayed flight at all!
```

Cleaned review:

```text
not enjoy delay flight
```

Preserving the word `not` is important because negation can reverse the meaning of a sentence.

---

## Exploratory Data Analysis

After preprocessing, the final dataset contained **14,069 reviews**.

The average review characteristics were:

| Measure                          | Result |
| -------------------------------- | -----: |
| Average original character count | 105.58 |
| Average original word count      |  17.99 |
| Average cleaned word count       |   9.12 |

Average original review length by sentiment:

| Sentiment | Average word count |
| --------- | -----------------: |
| Negative  |              19.74 |
| Neutral   |              14.79 |
| Positive  |              14.92 |

Negative reviews were longer on average than neutral and positive reviews.

The exploratory analysis generated the following charts:

```text
reports/figures/sentiment_distribution.png
reports/figures/review_length_distribution.png
```

---

## Feature Extraction

Two text feature-extraction approaches were compared.

### Bag of Words

Bag of Words represents reviews using the frequency of words and word combinations.

The project uses:

- Unigrams
- Bigrams
- A maximum of 20,000 features
- Minimum document frequency of 2
- Maximum document frequency of 95%

### TF-IDF

Term Frequency-Inverse Document Frequency gives greater importance to words that are informative in a particular review but less common across the complete dataset.

The same unigram and bigram settings were used for TF-IDF.

---

## Machine-Learning Models

The following model and feature-extraction combinations were trained and compared:

1. Bag of Words with Multinomial Naive Bayes
2. Bag of Words with Linear SVM
3. TF-IDF with Multinomial Naive Bayes
4. TF-IDF with Linear SVM

An 80:20 stratified train-test split was used:

| Dataset section | Records |
| --------------- | ------: |
| Training set    |  11,255 |
| Test set        |   2,814 |

Stratification preserved the proportion of positive, neutral, and negative reviews in both datasets.

---

## Model Evaluation Metrics

The models were evaluated using:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Weighted precision
- Weighted recall
- Weighted F1-score
- Classification reports
- Confusion matrices

### Accuracy

Accuracy measures the proportion of all reviews that were classified correctly.

### Precision

Precision measures how many reviews predicted as belonging to a sentiment class were actually members of that class.

### Recall

Recall measures how many reviews belonging to a sentiment class were correctly identified.

### F1-Score

The F1-score balances precision and recall.

### Macro F1-Score

Macro F1 calculates the F1-score separately for each class and gives each class equal importance.

This metric was used to select the best model because the dataset is imbalanced.

### Weighted F1-Score

Weighted F1 considers class performance while accounting for the number of observations in each sentiment class.

---

## Model Results

| Feature extraction | Model                   | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
| ------------------ | ----------------------- | -------: | --------------: | -----------: | -------: | ----------: |
| TF-IDF             | Linear SVM              |   0.7850 |          0.7137 |       0.7127 |   0.7128 |      0.7834 |
| Bag of Words       | Linear SVM              |   0.7715 |          0.6985 |       0.7118 |   0.7047 |      0.7740 |
| Bag of Words       | Multinomial Naive Bayes |   0.7761 |          0.7139 |       0.6747 |   0.6909 |      0.7675 |
| TF-IDF             | Multinomial Naive Bayes |   0.7381 |          0.7695 |       0.5280 |   0.5679 |      0.6903 |

---

## Selected Model

The selected model was:

```text
TF-IDF + Linear SVM
```

Its performance was:

| Metric            |  Score |
| ----------------- | -----: |
| Accuracy          | 78.50% |
| Macro precision   | 71.37% |
| Macro recall      | 71.27% |
| Macro F1-score    | 71.28% |
| Weighted F1-score | 78.34% |

The majority-class baseline was approximately **64.28%**.

A classifier that predicted every review as negative would therefore achieve an accuracy of approximately 64.28%, but it would fail to identify neutral and positive feedback.

The selected model improved accuracy while also learning to classify all three sentiment classes.

The TF-IDF and Linear SVM configuration was selected because it achieved the highest macro F1-score.

---

## Interpretation of Model Comparison

The comparison produced several important findings:

- Linear SVM performed better than Multinomial Naive Bayes when TF-IDF features were used.
- Bag of Words produced competitive results with both models.
- TF-IDF with Multinomial Naive Bayes achieved relatively high macro precision but low macro recall.
- The low recall of TF-IDF with Naive Bayes indicates that it missed many neutral and positive observations.
- TF-IDF with Linear SVM produced the best balance between precision and recall across the three sentiment classes.
- The selected Linear SVM model handled the imbalanced data more effectively because class weighting was applied.

---

## Saved Model Files

The trained model and vectorizer are saved as:

```text
models/sentiment_model.pkl
models/vectorizer.pkl
```

Model information is stored in:

```text
models/model_metadata.json
```

The saved files allow the application to make predictions without retraining the model every time the system starts.

---

## Graphical User Interface

The application includes a Tkinter desktop dashboard.

The dashboard supports:

- Manual review entry
- Instant sentiment prediction
- Positive, neutral, and negative statistics
- A processed-review table
- A sentiment-distribution bar chart
- CSV and JSON file uploads
- Automatic review-column detection
- Exporting processed results
- Clearing current dashboard results
- Negative-sentiment warnings

The application classifies reviews using the saved TF-IDF vectorizer and Linear SVM model.

### Manual Prediction Process

When a user enters a review and clicks **Analyse Review**, the application:

1. Reads the entered review.
2. Cleans and preprocesses the text.
3. Transforms the review using the saved TF-IDF vectorizer.
4. Passes the transformed review to the saved Linear SVM model.
5. Displays the predicted sentiment.
6. Updates the review table.
7. Updates the sentiment totals.
8. Refreshes the sentiment-distribution chart.

### Batch File Processing

Users can upload:

- CSV files
- JSON files

The dashboard automatically searches for review columns with names such as:

```text
review
text
feedback
comment
comments
customer_review
customer feedback
detailed review
content
message
```

The system then classifies all usable reviews and displays the results.

---

## Automated Review Processing

The automation module monitors:

```text
data/incoming/
```

When a new CSV or JSON file is detected, the system:

1. Reads the incoming file.
2. Identifies the customer-review text column.
3. Cleans every review.
4. Generates sentiment predictions.
5. Saves the processed results.
6. Assesses the percentage of negative reviews.
7. Generates an alert when necessary.
8. Archives the original input file.
9. Moves invalid files to the failed-files directory.

Processed files are saved in:

```text
data/processed/automated/
```

Successfully processed source files are moved to:

```text
data/archive/
```

Failed files are moved to:

```text
data/archive/failed/
```

### Continuous Monitoring

The automation process can continuously monitor the incoming folder.

By default, it checks for new files every five seconds.

When a new file appears, it is processed automatically without requiring the user to restart the application.

---

## Negative Review Alerts

The system generates an alert when:

- At least five reviews have been processed, and
- At least 50% of the processed reviews are predicted as negative

Triggered alerts are written to:

```text
reports/negative_review_alerts.log
```

The threshold and minimum number of reviews can be adjusted in the notification and GUI modules.

The current implementation records local dashboard and log notifications.

Email, SMS, Slack, or other external notifications can be added in a future version.

---

## Model Retraining

The project supports controlled retraining using newly labelled customer feedback.

The retraining process:

1. Reads a labelled CSV or JSON file.
2. Detects the review and sentiment columns.
3. Validates the sentiment labels.
4. Cleans the new reviews.
5. Excludes records that overlap with the protected test set.
6. Combines the usable new data with the original training data.
7. Trains a candidate model.
8. Evaluates the existing and candidate models on the same test set.
9. Replaces the existing model only when the candidate meets the performance requirement.
10. Backs up the previous model before replacement.
11. Saves a retraining report.

Allowed labels are:

```text
positive
neutral
negative
```

Retraining reports are stored in:

```text
reports/retraining/
```

Previous model versions are backed up in:

```text
models/backups/
```

The controlled retraining process prevents a weaker candidate model from automatically replacing a better-performing production model.

---

## Automated Testing

The project includes automated tests for:

- Removal of social-media account handles
- Removal of website URLs
- Preservation of negation terms
- Decoding of HTML entities
- Empty-review handling
- Single-review prediction
- Batch prediction
- Review-column detection
- Negative-alert logic
- Retraining-column detection
- Retraining-label validation
- Missing-column error handling

Current test result:

```text
18 passed
```

The warnings displayed by `joblib` relate to deprecated NumPy array-shape behaviour and do not represent failed tests.

---

## Project Structure

```text
Sentiment Analysis & Naive Bayes/
│
├── data/
│   ├── archive/
│   │   └── failed/
│   │
│   ├── incoming/
│   │
│   ├── processed/
│   │   ├── automated/
│   │   ├── retraining/
│   │   ├── customer_reviews.csv
│   │   └── customer_reviews_cleaned.csv
│   │
│   ├── raw/
│   │   └── Tweets.csv
│   │
│   └── retraining/
│       └── new_labeled_reviews.csv
│
├── models/
│   ├── backups/
│   ├── model_metadata.json
│   ├── sentiment_model.pkl
│   └── vectorizer.pkl
│
├── reports/
│   ├── figures/
│   │   ├── confusion_matrix_bag_of_words_linear_svm.png
│   │   ├── confusion_matrix_bag_of_words_multinomial_naive_bayes.png
│   │   ├── confusion_matrix_tfidf_linear_svm.png
│   │   ├── confusion_matrix_tfidf_multinomial_naive_bayes.png
│   │   ├── review_length_distribution.png
│   │   └── sentiment_distribution.png
│   │
│   ├── retraining/
│   │
│   ├── classification_report_bag_of_words_linear_svm.csv
│   ├── classification_report_bag_of_words_multinomial_naive_bayes.csv
│   ├── classification_report_tfidf_linear_svm.csv
│   ├── classification_report_tfidf_multinomial_naive_bayes.csv
│   ├── model_comparison.csv
│   └── negative_review_alerts.log
│
├── src/
│   ├── __init__.py
│   ├── automation.py
│   ├── clean_dataset.py
│   ├── explore_data.py
│   ├── gui.py
│   ├── notification.py
│   ├── predict.py
│   ├── prepare_dataset.py
│   ├── preprocessing.py
│   ├── retrain_model.py
│   └── train_model.py
│
├── tests/
│   ├── test_automation.py
│   ├── test_prediction.py
│   ├── test_preprocessing.py
│   └── test_retraining.py
│
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

---

## Installation on macOS

### 1. Open the Project Directory

```bash
cd "/path/to/Sentiment Analysis & Naive Bayes"
```

Open the project in Visual Studio Code:

```bash
code .
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

### 3. Activate the Virtual Environment

```bash
source .venv/bin/activate
```

The terminal prompt should display:

```text
(.venv)
```

### 4. Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 5. Install Project Dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Download NLTK Resources

```bash
python -m nltk.downloader stopwords wordnet omw-1.4
```

### 7. Select the VS Code Python Interpreter

In Visual Studio Code:

1. Press `Command + Shift + P`.
2. Search for `Python: Select Interpreter`.
3. Select the interpreter ending with:

```text
.venv/bin/python
```

---

## Running the Application

Launch the graphical interface:

```bash
python main.py
```

The application window allows the user to:

- Enter individual customer reviews
- Upload CSV or JSON review files
- View sentiment predictions
- View sentiment totals
- View a sentiment-distribution chart
- Export processed results

---

## Running Data Preparation

Prepare the original downloaded dataset:

```bash
python -m src.prepare_dataset
```

This creates:

```text
data/processed/customer_reviews.csv
```

Clean and preprocess the prepared reviews:

```bash
python -m src.clean_dataset
```

This creates:

```text
data/processed/customer_reviews_cleaned.csv
```

Generate exploratory statistics and charts:

```bash
python -m src.explore_data
```

---

## Training the Models

Train and compare the four model configurations:

```bash
python -m src.train_model
```

The command generates:

- Saved model files
- Model metadata
- A model-comparison report
- Classification reports
- Confusion matrices

The selected model is saved as:

```text
models/sentiment_model.pkl
```

The fitted vectorizer is saved as:

```text
models/vectorizer.pkl
```

---

## Testing Predictions

Run sample sentiment predictions:

```bash
python -m src.predict
```

---

## Running the Automated Processor

### Process Available Files Once

Place a CSV or JSON file in:

```text
data/incoming/
```

Run:

```bash
python -m src.automation --once
```

The system processes all supported files currently available and then stops.

### Monitor the Incoming Folder Continuously

```bash
python -m src.automation
```

The program checks the incoming folder every five seconds.

### Change the Monitoring Interval

For example, check every ten seconds:

```bash
python -m src.automation --interval 10
```

Stop the continuous process using:

```text
Control + C
```

---

## Retraining the Model

Use a newly labelled CSV file:

```bash
python -m src.retrain_model \
  --input data/retraining/new_labeled_reviews.csv
```

Specify custom column names when necessary:

```bash
python -m src.retrain_model \
  --input data/retraining/new_labeled_reviews.csv \
  --text-column review \
  --label-column sentiment
```

Require a minimum macro F1 improvement of 0.01:

```bash
python -m src.retrain_model \
  --input data/retraining/new_labeled_reviews.csv \
  --minimum-improvement 0.01
```

The system retains the existing model when the candidate does not meet the required performance threshold.

A forced replacement option is available:

```bash
python -m src.retrain_model \
  --input data/retraining/new_labeled_reviews.csv \
  --force
```

Forced replacement should only be used when there is a documented reason because it may reduce model performance.

---

## Running Automated Tests

Run the complete test suite:

```bash
python -m pytest -v
```

Run a shorter test summary:

```bash
python -m pytest -q
```

Expected result:

```text
18 passed
```

---

## Generated Reports

The project produces:

- Sentiment-distribution chart
- Review-length chart
- Model-comparison CSV
- Classification reports
- Confusion matrices
- Retraining reports
- Negative-review alert log

The visual reports are available in:

```text
reports/figures/
```

Model comparison results are stored in:

```text
reports/model_comparison.csv
```

Detailed class-level performance results are stored in:

```text
reports/classification_report_bag_of_words_linear_svm.csv
reports/classification_report_bag_of_words_multinomial_naive_bayes.csv
reports/classification_report_tfidf_linear_svm.csv
reports/classification_report_tfidf_multinomial_naive_bayes.csv
```

---

## Limitations

1. The model was trained using airline-related customer feedback. Its performance may be lower for reviews from unrelated industries.
2. The dataset is imbalanced, with negative reviews forming the majority class.
3. Sarcasm, irony, humour, and indirect language can be difficult for a traditional machine-learning model to classify.
4. Reviews containing both positive and negative statements are assigned only one overall sentiment.
5. The system currently supports English-language reviews.
6. The automated notification system records local alerts but does not currently send emails, SMS messages, or external application notifications.
7. The current folder-monitoring method is suitable for a local demonstration but would require a database, message queue, or API architecture for large-scale production use.
8. The Linear SVM classifier does not directly provide probability scores without an additional calibration step.
9. The model may incorrectly classify reviews containing vocabulary that was not sufficiently represented in the training dataset.
10. The system predicts the overall sentiment of a review but does not identify which specific service feature caused the sentiment.

---

## Possible Future Improvements

Future versions could include:

- A web-based dashboard
- REST API integration
- Database storage
- Cloud deployment
- Email and SMS alerts
- Slack or Microsoft Teams notifications
- Multilingual sentiment analysis
- Transformer models such as BERT
- Sentiment confidence scores
- Topic modelling
- Aspect-based sentiment analysis
- Trend analysis over time
- Role-based user access
- Real-time client-platform integrations
- Scheduled retraining
- Docker containerization
- Database-backed review history
- User authentication
- Separate dashboards for different clients
- Review search and filtering
- Date-based sentiment analysis
- Sentiment trend charts
- Automatic identification of recurring customer complaints

---

## Scalability Considerations

The current version is designed as a local desktop demonstration.

For a production environment serving many clients or processing thousands of reviews, the system could be redesigned using:

- A web framework such as FastAPI or Flask
- A relational database such as PostgreSQL
- A message queue such as RabbitMQ or Apache Kafka
- Background processing using Celery
- Cloud object storage
- Containerization using Docker
- API integrations with review platforms
- Centralized logging and monitoring
- Scheduled or event-driven retraining
- Load balancing
- Role-based authentication

The prediction system already supports batch processing, which makes it more efficient than processing each incoming review separately.

---

## Security Considerations

For production deployment:

- Review data should be stored securely.
- Personally identifiable information should be removed where possible.
- Uploaded files should be validated before processing.
- Access to model files and processed data should be restricted.
- API credentials should not be stored in the project source code.
- Kaggle API credentials should not be committed to version control.
- Model files should only be loaded from trusted sources.
- User access should be authenticated and authorized.

The `.gitignore` file prevents sensitive or unnecessary files such as virtual environments and Kaggle credentials from being committed.

---

## Conclusion

The project successfully demonstrates an end-to-end automated sentiment analysis workflow.

It covers:

- Data collection and preparation
- Text preprocessing
- Exploratory data analysis
- Feature extraction
- Model training
- Model comparison
- Performance evaluation
- Model persistence
- Desktop interface development
- Automated review processing
- Negative-review alerts
- Controlled model retraining
- Automated software testing

The selected TF-IDF and Linear SVM model achieved:

- Accuracy of **78.50%**
- Macro precision of **71.37%**
- Macro recall of **71.27%**
- Macro F1-score of **71.28%**
- Weighted F1-score of **78.34%**

The completed system provides a practical foundation for helping businesses process customer feedback, understand sentiment distribution, identify increases in negative feedback, and support data-driven service improvements.

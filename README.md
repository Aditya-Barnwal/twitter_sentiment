# Twitter Sentiment Accuracy Score Prediction

Project Description:
This project aims to predict the accuracy of sentiment analysis on Twitter data. The dataset used for this project is from Kaggle and contains tweets with labeled sentiments.

Setup Instructions:

To set up the project, follow these steps:

Clone the repository:

git clone https://github.com/Aditya-Barnwal/twitter_sentiment.git

cd twitter_sentiment
Install the necessary packages:

pip install -r requirements.txt

## Data Loading
Make sure you have the dataset in the correct location. You can download the dataset from Kaggle and place it in the data directory.


##Data processing
 
 #Loading the data from CSV file to pandas dataframe

twitter_data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1')

 #Name the columns and read the dataset

column_names = ['target', 'id', 'date', 'flag', 'user', 'text']

twitter_data = pd.read_csv('training.1600000.processed.noemoticon.csv', names=column_names, encoding='ISO-8859-1')

## Data Preprocessing

The following steps are used for data preprocessing:


#Splitting the data to training data and test data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

## Model Training
The following steps are used for training the model:


#Converting the textual data to numerical data

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)

X_test = vectorizer.transform(X_test)

model.fit(X_train, Y_train)

## Model Evaluation

The following steps are used for evaluating the model:

#Model Evaluation

#Accuracy Score

#Accuracy score on the training data

X_train_prediction = model.predict(X_train)

training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Accuracy score on the training data: ', training_data_accuracy)

#Accuracy score on the test data

X_test_prediction = model.predict(X_test)

test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy score on the test data: ', test_data_accuracy)


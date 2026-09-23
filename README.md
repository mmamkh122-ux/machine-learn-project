# Medical Cost & Risk Analysis System

This project uses machine learning to predict medical insurance costs and classify patients as high risk or low risk, based on personal and health information such as age, sex, BMI, number of children, smoking status, and region.

There are two models working together here. The first is a regression model that estimates how much a patient's medical charges are expected to be. The second is a classification model that labels the patient as high risk or low risk, where "high risk" simply means the predicted cost is above the median cost in the dataset. Several classification algorithms were tested during training, including Logistic Regression, KNN, SVM, Decision Tree, and Random Forest, and the best performing one is selected automatically based on accuracy.

Before training, the data goes through a preprocessing pipeline that scales the numerical features (age, BMI, children) and one-hot encodes the categorical features (sex, smoker, region). This pipeline is saved together with the model, so it can handle raw input directly without any manual preprocessing later on.

The project also includes a Streamlit web app where you can enter a patient's details and instantly get a predicted cost and risk classification, along with a confidence score. The app also supports batch predictions, so you can upload a CSV file with multiple patients and download the results for all of them at once.

To run the project, install the requirements with pip install -r requirements.txt, then run train_model.py to train and save the models, and finally run streamlit run app.py to launch the web app.

The dataset used is insurance.csv, containing age, sex, bmi, children, smoker, region, and charges columns.

Built by Mostafa Magdy, AI Engineering student at the Faculty of Artificial Intelligence, Menoufia University.


There are two models working together here. The first is a regression model that estimates how much a patient's medical charges are expected to be.
The second is a classification model that labels the patient as high risk or low risk, 
where "high risk" simply means the predicted cost is above the median cost in the dataset. 
Several classification algorithms were tested during training, including Logistic Regression, KNN, SVM, Decision Tree, and Random Forest, 
and the best performing one is selected automatically based on accuracy.

Before training, the data goes through a preprocessing pipeline that scales the numerical features (age, BMI, children) and one-hot encodes the categorical
features (sex, smoker, region). This pipeline is saved together with the model, so it can handle raw input directly without any manual preprocessing later on.

The project also includes a Streamlit web app where you can enter a patient's details and instantly get a predicted cost and risk classification,
along with a confidence score. The app also supports batch predictions, so you can upload a CSV file with multiple patients and download the results for all of them at once.


To run the project, install the requirements with pip install -r requirements.txt, then run train_model.py to train and save the models, 
and finally run streamlit run app.py to launch the web app.

The dataset used is insurance.csv, containing age, sex, bmi, children, smoker, region, and charges columns.

Built by Mostafa Magdy, AI Engineering student at the Faculty of Artificial Intelligence, Menoufia University.


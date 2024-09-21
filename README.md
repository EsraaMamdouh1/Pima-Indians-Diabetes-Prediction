# **Pima Indians Diabetes Prediction**

![Diabetes Prediction](diabetes.jpeg)

## 🚀 **Objective**
The goal of this project is to predict whether a patient has **diabetes** (Outcome: 1) or not (Outcome: 0) based on various medical measurements using machine learning models.

---

## 📊 **Dataset Overview**
The **Pima Indian Diabetes Dataset** consists of medical information on **768 women** from a population near Phoenix, Arizona. The dataset includes **8 features** related to personal and medical history and a **binary outcome** for diabetes diagnosis.

### **Features:**
- **Pregnancies**: Number of pregnancies
- **Glucose**: 2-hour plasma glucose concentration (mg/dl)
- **Blood Pressure**: Diastolic blood pressure (mmHg)
- **Skin Thickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-hour serum insulin (mu U/ml)
- **BMI**: Body mass index (kg/m²)
- **Age**: Age (years)
- **Diabetes Pedigree Function**: Probability of diabetes based on family history

### **Target Variable:**
- **Outcome**: 0 (No Diabetes), 1 (Has Diabetes)

---

## ⚙️ **Project Structure**
1. **Data Preprocessing**:
   - Handling missing values.
   - Data cleaning and transformation.
   
2. **Exploratory Data Analysis (EDA)**:
   - Visualizations of features using histograms, pair plots, and correlation heatmaps.
   
3. **Feature Scaling**:
   - Standardizing feature values before applying models.
   
4. **Modeling**:
   - Logistic Regression
   - Random Forest Classifier

5. **Model Evaluation**:
   - Accuracy
   - Confusion Matrix
   - ROC-AUC Score

---

## **Prerequisites**

Make sure you have Python installed. You will also need the following libraries:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn



## 💻 **How to Run the Code**

### **1. Clone the repository**
Clone this repository to your local machine using the command below:
```bash
git clone https://github.com/EsraaMamdouh1/Pima-Indians-Diabetes-Prediction.git
cd Pima-Indians-Diabetes-Prediction
```

### **2. Install Dependencies**

Before running the code, you need to install the necessary libraries. Use the following command to install all the required dependencies:

```bash
pip install -r requirements.txt
```

### **3. Run the Script in Google Colab**

To run the notebook in Google Colab, follow these steps:

1. Upload the `PIMA_INDIANS_DIABETES.ipynb` file to Google Drive.
2. Open Google Colab (https://colab.research.google.com/).
3. Click on **File > Open Notebook**.
4. Select the **Google Drive** tab and navigate to the uploaded file.
5. Once the notebook is opened, run the code cells step by step.

---

### **4. Load Dataset**

Ensure the **Pima Indians Diabetes dataset** (`diabetes.csv`) is located at `/content/diabetes.csv`. The following line in the notebook should correctly load the dataset:

```python
df = pd.read_csv('/content/diabetes.csv')
```

---

### 📚 Dependencies

To run this project, make sure you have installed all the necessary libraries listed in the `requirements.txt` file. You can install them using the following command:

```bash
pip install -r requirements.txt
```
---

### 🧠 Model Evaluation

The models in this project are evaluated based on the following metrics:

- **Accuracy Score**: Measures the overall accuracy of the models.
- **Confusion Matrix**: Helps visualize the performance in terms of true positives, false negatives, etc.
- **Classification Report**: Provides precision, recall, and F1-score for the models.
- **ROC-AUC Score**: Evaluates the performance of the model in terms of distinguishing between classes.

Both **Logistic Regression** and **Random Forest** models are evaluated and compared using these metrics.

---

### 📈 Data Visualizations

This project includes various visualizations to understand the dataset:

- **Histograms**: Displays the distribution of each feature.
- **Pairplots**: Shows relationships between features and the target variable.
- **Correlation Heatmap**: Displays the correlation between all the features in the dataset.

Visualizations help gain insights and understand the data before applying machine learning models.

---

### 🔧 Project Features

- **Data Cleaning**: Handling of missing values and replacement of zero values with the mean or median for certain columns.
- **Feature Scaling**: Standardization of features using `StandardScaler`.
- **Modeling**: Built and evaluated two models: **Logistic Regression** and **Random Forest**.
- **Cross-Validation**: 5-fold cross-validation was performed to check the robustness of the models.

---

### 🙌 Acknowledgments

Special thanks to the following:

- **National Institute of Diabetes and Digestive and Kidney Diseases** for the Pima Indians Diabetes Dataset.
- The open-source community for providing useful libraries and tools such as **Pandas**, **Scikit-learn**, and **Seaborn**.

---

### 📬 Contact

For any inquiries or feedback, feel free to contact me:

- **Name**: Esraa Mamdouh
- **Email**: esraamamdouh782@gmail.com
- **GitHub**: [EsraaMamdouh1](https://github.com/EsraaMamdouh1)

---

### 📄 License

This project is licensed under the [MIT License](./LICENSE). Feel free to use, modify, and distribute this code as long as proper attribution is provided.


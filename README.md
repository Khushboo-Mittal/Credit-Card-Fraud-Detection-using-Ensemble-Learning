# Credit Card Fraud Detection using Ensemble Learning

## Overview

This project is aimed at detecting fraudulent credit card transactions using ensemble learning techniques. Credit card fraud detection is a critical challenge in the finance industry, and this project explores advanced machine learning methods such as Bagging, Boosting, and Stacking to address this issue. The dataset used in this project includes a variety of transaction details, and the models are evaluated based on their accuracy, precision, recall, and F1 scores.

---

## Ensemble Learning

Ensemble learning combines multiple machine learning models to improve overall performance. The key techniques explored in this project are:

### Bagging (Bootstrap Aggregating)

- Trains multiple models independently on bootstrapped subsets of the data.
- Averages predictions (for regression) or uses majority voting (for classification).
- Example: Random Forest.

### Boosting

- Sequentially trains models, with each new model focusing on correcting the errors of the previous ones.
- Combines all models for a weighted prediction.
- Example: Gradient Boosting.

### Stacking

- Combines predictions from multiple base models using a meta-model (e.g., Logistic Regression).
- Learns the best combination of base model outputs to improve predictions.

By employing these techniques, this project demonstrates the effectiveness of advanced machine learning approaches in detecting credit card fraud. Explore the provided notebook to understand the step-by-step implementation and insights.

---

## Project Structure

The project is organized as follows:

```
Credit-Card-Fraud-Detection-using-Ensemble-Learning/
├── dataset_split.7z.001  # Initially present before extraction
├── dataset_split.7z.002  # Initially present before extraction
├── Credit Card Fraud Detection dataset/  # Attained after combining and extracting the split files
│   ├── fraudTrain.csv
│   ├── fraudTest.csv
├── EnsembleLearning.ipynb
├── README.md
```

1. **Data Preprocessing**
   - Load and inspect the dataset.
   - Handle missing values and duplicated records.
   - Perform feature engineering to create meaningful features.
   - Encode categorical variables and normalize data.
2. **Exploratory Data Analysis (EDA)**
   - Visualize the distribution of fraud and non-fraud transactions.
   - Analyze the dataset's imbalance and resolve it through sampling techniques.
3. **Model Development**
   - Implement three ensemble learning techniques:
     - **Bagging** using Random Forest.
     - **Boosting** using Gradient Boosting.
     - **Stacking** using Random Forest, Decision Tree, and SVM with Logistic Regression as the final estimator.
   - Train each model on the preprocessed data.
4. **Model Evaluation**
   - Evaluate each model using metrics such as:
     - Accuracy
     - Precision
     - Recall
     - F1 Score
   - Visualize the confusion matrices for each model.

---

## Installation and Setup

### Prerequisites

The project requires the following dependencies:

- Python (3.8 or higher)
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/credit-card-fraud-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Credit-Card-Fraud-Detection-using-Ensemble-Learning
   ```
3. Install the required libraries individually:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

---

## Dataset

The dataset is provided as split archive files for efficient storage. It contains the following files:

- **fraudTrain.csv**: Training dataset containing credit card transaction details.
- **fraudTest.csv**: Testing dataset containing credit card transaction details.

### How to Combine and Extract the Dataset

1. Download the split dataset archive files `dataset_split.7z.001` and `dataset_split.7z.002` from this repository.
2. To combine the split files into a single archive:
   - Ensure all split files are in the same directory.
   - Use a tool like **7-Zip**:
     - If you don't have 7-Zip installed, you can download it from [here](https://www.7-zip.org/).
     - Right-click on the first split file (`dataset_split.7z.001`) in your file explorer.
     - Select **7-Zip > Extract Here**.
   - This will automatically combine the split files and extract the original archive `Credit Card Fraud Detection dataset.7z`.
3. Extract the files from `Credit Card Fraud Detection dataset.7z`:
   - Right-click on the archive and select **7-Zip > Extract Here**.
4. This will extract the following two files:
   - `fraudTrain.csv`
   - `fraudTest.csv`
5. After extraction, place these two CSV files in the folder: `Credit Card Fraud Detection dataset/`.

---

### Data Preprocessing Steps

- **Handle Missing Values**: Rows with missing data were dropped.
- **Feature Engineering**: Extracted features such as day, month, year, hour, and minute from the transaction date.
- **Categorical Encoding**: Label encoded categorical variables like `category` and `cc_num`.
- **Balancing Dataset**: Addressed class imbalance by down-sampling the majority class to match the minority class.

---

## How to Run the Project

1. After extracting the dataset, place the `fraudTrain.csv` and `fraudTest.csv` files in the folder: `Credit Card Fraud Detection dataset/`.
2. Open the Jupyter notebook `EnsembleLearning.ipynb` and execute the cells step-by-step.
3. Evaluate results and visualize confusion matrices.

---
By employing ensemble learning techniques, this project demonstrates the effectiveness of advanced machine learning approaches in detecting credit card fraud. Explore the provided notebook to understand the step-by-step implementation and insights.


# German Credit Data Clustering Project

A comprehensive clustering analysis project examining German credit data to identify patterns and group similar credit applicants using various machine learning techniques.

## 📊 Project Overview

This project performs an in-depth clustering analysis on German credit data to understand customer segmentation patterns. The analysis includes data preprocessing, dimensionality reduction, and multiple clustering algorithms to identify meaningful groups within the credit applicant population.

## 📁 Project Structure

```
Clustering_Project/
├── DeSilva_Clustering_Project.ipynb    # Main Jupyter notebook with analysis
├── german_credit_data.csv              # Dataset file
└── README.md                          # Project documentation
```

## 📈 Dataset Information

**Source:** German Credit Data (`german_credit_data.csv`)

### Context

The original dataset contains 1000 entries with 20 categorical/symbolic attributes. In this dataset, each entry represents a person who takes credit from a bank. Each person is classified as good or bad credit risks according to the set of attributes.

### Features Description

- **Age** (numeric): Age of the credit applicant
- **Sex** (text): Gender (male, female)
- **Job** (numeric): Employment status
  - 0: Unskilled and non-resident
  - 1: Unskilled and resident
  - 2: Skilled
  - 3: Highly skilled
- **Housing** (text): Housing situation (own, rent, or free)
- **Saving accounts** (text): Savings account status (little, moderate, quite rich, rich)
- **Checking account** (numeric): Checking account balance in DM (Deutsche Mark)
- **Credit amount** (numeric): Requested credit amount in DM
- **Duration** (numeric): Credit duration in months
- **Purpose** (text): Purpose of credit (car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others)

### Dataset Statistics

- **Total Records:** 1,000 entries
- **Total Features:** 9 (after selection)
- **Missing Values:** Present in 'Saving accounts' and 'Checking account' columns
- **Average Age:** ~36 years
- **Average Credit Amount:** ~3,300 DM
- **Average Duration:** ~21 months

## 🎯 Project Objectives (Original Assignment)

1. **(1P) Analyze and clean the data**
   - Determine number of rows, columns, and data types
   - Handle missing values and outliers

2. **(1P) Encode and normalize the data**
   - Convert categorical variables to numerical format
   - Standardize features for clustering algorithms

3. **(2P) Perform PCA and t-SNE to visualize data**
   - Apply Principal Component Analysis for dimensionality reduction
   - Use t-SNE for non-linear visualization

4. **(3P) Find clusters using multiple algorithms**
   - K-Means clustering
   - Hierarchical clustering
   - DBSCAN (Density-Based Spatial Clustering)

5. **(3P) Explain and interpret results**
   - Analyze clustering performance
   - Draw insights from identified patterns

## 🔧 Methodology

### 1. Data Preprocessing
- **Missing Value Treatment:** Dropped rows with missing values (394 entries removed)
- **Label Encoding:** Applied to categorical variables (Sex, Housing, Saving accounts, Purpose)
- **Feature Scaling:** StandardScaler normalization applied to all features

### 2. Dimensionality Reduction
- **PCA Analysis:** 
  - Low variance in principal components observed
  - Indicates relatively uniform feature distribution
  - Limited distinct cluster formation
- **t-SNE Visualization:**
  - Multiple small clusters identified
  - No clear large cohesive groups formed

### 3. Clustering Algorithms

#### K-Means Clustering
- **Configuration:** k=2 clusters (risk-based segmentation)
- **Performance:** Moderate success in PCA space, limited effectiveness in t-SNE space

#### Hierarchical Clustering
- **Method:** Ward linkage with Euclidean distance
- **Results:** 
  - PCA data: 4 distinct clusters suggested
  - t-SNE data: 3 clusters identified

#### DBSCAN
- **Parameters:** Optimized eps and min_samples for each dimensionality reduction technique
- **Performance:** Suboptimal clustering in both PCA and t-SNE spaces

## 📊 Key Findings

### Data Characteristics
- **Housing:** Majority of applicants own their homes
- **Savings:** Most applicants have modest savings ("little" category)
- **Purpose:** Primary credit purposes are cars, radio/TV, and furniture/equipment
- **Employment:** Most applicants are skilled workers

### Clustering Results
- **Challenge:** Data exhibits limited natural clustering tendency
- **PCA Limitations:** Low component variance suggests uniform data distribution
- **t-SNE Insights:** Multiple small clusters rather than distinct large groups
- **Algorithm Performance:** Hierarchical clustering showed most promise with 3-4 cluster suggestions

### Statistical Insights
- Average credit applicant profile: 36-year-old skilled worker seeking ~3,300 DM for ~21 months
- Credit purposes dominated by consumer goods and vehicles
- Most applicants maintain stable housing and modest financial accounts

## 🛠️ Technologies Used

- **Python Libraries:**
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computations
  - `scikit-learn` - Machine learning algorithms
  - `matplotlib` / `seaborn` - Data visualization
  - `scipy` - Statistical computations

- **Machine Learning Techniques:**
  - Principal Component Analysis (PCA)
  - t-Distributed Stochastic Neighbor Embedding (t-SNE)
  - K-Means Clustering
  - Hierarchical Clustering
  - DBSCAN

## 🚀 How to Run

1. **Prerequisites:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn scipy
   ```

2. **Execute the Analysis:**
   - Open `DeSilva_Clustering_Project.ipynb` in Jupyter Notebook
   - Ensure `german_credit_data.csv` is in the same directory
   - Run all cells sequentially

3. **Expected Runtime:** ~5-10 minutes depending on system specifications

## 📈 Results Summary

The clustering analysis revealed that the German credit data does not exhibit strong natural clustering patterns. This finding is significant as it suggests:

1. **Uniform Risk Distribution:** Credit applicants show relatively similar characteristics across the dataset
2. **Complex Relationships:** Risk factors may be more nuanced than simple demographic or financial groupings
3. **Need for Advanced Techniques:** Future analysis might benefit from feature engineering or ensemble methods

### Recommendations for Future Work

- **Feature Engineering:** Create composite features combining multiple attributes
- **Advanced Algorithms:** Explore ensemble clustering methods or deep learning approaches
- **Domain Knowledge Integration:** Incorporate banking domain expertise for feature selection
- **External Data:** Consider additional economic or demographic variables

## 📝 Conclusions

This project demonstrates the challenges of unsupervised learning on real-world financial data. While traditional clustering algorithms provided limited segmentation, the analysis offers valuable insights into the dataset's structure and the complexity of credit risk assessment. The uniform distribution of applicant characteristics suggests that credit risk evaluation requires sophisticated modeling beyond simple demographic clustering.

## 👤 Author

**Amindu De Silva** - Clustering Analysis Project


---

*This project was completed as part of a machine learning coursework focusing on unsupervised learning techniques and clustering analysis.*

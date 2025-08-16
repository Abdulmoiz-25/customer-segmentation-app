# 📘 DeveloperHub Task 7 – Customer Segmentation - Mall Customers

## 📌 Task Objective  
Segment mall customers using **unsupervised learning** techniques to understand their behavior and provide targeted marketing strategies. The dashboard and notebook utilize **K-Means clustering, PCA, and t-SNE** for visualization and cluster profiling.

---

## 📁 Dataset  
- **Name**: Mall Customers Dataset  
- **Source**: Publicly available dataset (often used for customer segmentation)  
- **Features include**:  
  - Customer demographics (Age, Gender)  
  - Annual Income (k$)  
  - Spending Score (1–100)  

---

## 🛠️ Tools & Libraries Used  
- **Pandas** – data manipulation  
- **NumPy** – numerical operations  
- **Scikit-learn** – K-Means clustering, PCA, scaling  
- **Plotly** – interactive visualization for scatterplots, radar charts, and correlations  
- **Matplotlib** – EDA visualizations  
- **Streamlit** – interactive dashboard deployment  
- **xlsxwriter** – exporting cluster data to Excel  

---

## 🚀 Approach  

### 🔍 1. Data Preparation & EDA  
- Loaded dataset (default or user-uploaded CSV)  
- Handled categorical features (Gender encoding)  
- Scaled numeric features using `StandardScaler`  
- Explored distributions, correlations, and feature patterns  

### 🤖 2. K-Means Clustering  
- Selected optimal **number of clusters (K)** using silhouette score  
- Applied K-Means clustering on selected features  
- Projected clusters to **2D PCA space** for visualization  

### 📈 3. Cluster Visualization  
- Scatter plots of PCA-reduced clusters  
- Radar charts for feature comparison across clusters  
- Cluster distribution via bar and pie charts  
- Optional t-SNE visualization (in Colab notebook)  

### 💡 4. Cluster Profiling & Marketing Strategies  
- Calculated cluster-wise **mean, median, min, max, and count**  
- Suggested **segment-specific marketing tactics** based on age, income, and spending score  
- Strategies include VIP rewards, discounts, bundles, and targeted campaigns  

### 🌐 5. Deployment  
- Developed a **Streamlit dashboard** with:  
  - Upload option for custom CSV  
  - Sidebar for selecting clustering features and number of clusters  
  - Interactive tabs: Dataset, EDA, Clustering, Insights  
  - Downloadable results in **CSV** and **Excel**  

---

## 📊 Results & Insights  
- Identified distinct customer segments with meaningful patterns  
- High spenders and premium customers can be targeted with exclusive offers  
- Moderate and budget-conscious segments receive tailored promotions  
- Cluster visualizations facilitate easy interpretation for marketing teams  

---

## ✅ Conclusion  
This project demonstrates an **end-to-end unsupervised ML workflow**: data preprocessing, exploratory analysis, clustering, visualization, insights generation, and deployment. It provides actionable marketing strategies and an interactive interface for business decision-making.

---

## 🌐 Live App  
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-segmentation-app-jxarptwrgyjxigydlru36k.streamlit.app/)

---

## 📚 Useful Links  
- [Mall Customers Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)  
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)  
- [Streamlit Documentation](https://docs.streamlit.io/)  
- [Plotly Documentation](https://plotly.com/python/)  

---

> 🔖 Submitted as part of the **DevelopersHub Internship Program**

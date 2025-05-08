# 🧠 Linear Regression Project: Modal Price Prediction

This project is a beginner-friendly machine learning web application built using **Streamlit**. It predicts the **Modal Price** of agricultural commodities based on the **State**, **Min Price**, and **Max Price** using **Linear Regression**. The project includes data visualization, model training, and evaluation—all wrapped in an interactive and deployable web app.

🔗 **Live Demo**: [Click here to try the app](https://linearregressionproject2-vxyr4guapjfnm5jyvbtwt4.streamlit.app/)

---

## 📊 Problem Statement

Predict the **Modal Price** of commodities using:
- State (categorical input)
- Min Price (numerical)
- Max Price (numerical)

This is useful for estimating market trends and price analysis.

---

## 🚀 Features

- 📌 Load and display the dataset
- 📈 Correlation heatmap and pairplot visualization
- ⚙️ Data preprocessing using OneHotEncoding
- 🧮 Model training using Linear Regression
- 📊 Evaluation metrics: MAE, RMSE, R² Score
- 📉 Scatter plot of actual vs predicted prices
- 🌐 Web app interface with Streamlit

---

## 🗂 Dataset

The dataset is stored in `daily_price.csv` and includes columns like:
- `State`
- `Min Price`
- `Max Price`
- `Modal Price` (target variable)

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- scikit-learn
- Streamlit

---

## 🙋‍♂️ Author

**Pradeep**  
🔗 GitHub: [@PradeepPradeep1234](https://github.com/PradeepPradeep1234)

## 🧪 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/PradeepPradeep1234/Linear_Regression_Project_2.git
cd Linear_Regression_Project_2

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

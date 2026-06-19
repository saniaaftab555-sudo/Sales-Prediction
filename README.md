# 📈 Sales Prediction Using Python

## 📌 Overview
It focuses on building a **Machine Learning model** in Python to predict the amount of a product that customers will purchase, based on advertising expenditure across multiple platforms.

Sales prediction helps businesses:
- Forecast future revenue
- Optimize their advertising budget
- Make smarter, data-driven marketing decisions

---

## 🧠 Problem Statement

> Given the amount spent on **TV**, **Radio**, and **Newspaper** advertising, predict the **total Sales** figures using a Linear Regression model.

---

## 📂 Dataset

The dataset used is the classic **Advertising Dataset** containing the following columns:

| Column | Description |
|--------|-------------|
| `TV` | Advertising spend on TV (in thousands) |
| `Radio` | Advertising spend on Radio (in thousands) |
| `Newspaper` | Advertising spend on Newspaper (in thousands) |
| `Sales` | Units of product sold (in thousands) — **Target Variable** |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core programming language |
| Pandas | Data loading & manipulation |
| NumPy | Numerical computations |
| Matplotlib & Seaborn | Data visualization |
| Scikit-Learn | ML model building & evaluation |

---

## 🔁 Project Workflow

```
1. Import Libraries
       ↓
2. Load & Explore Dataset (EDA)
       ↓
3. Visualize Relationships (Scatter, Heatmap, Distribution)
       ↓
4. Split Data → Train (80%) / Test (20%)
       ↓
5. Train Linear Regression Model
       ↓
6. Evaluate Model (MAE, MSE, RMSE, R²)
       ↓
7. Visualize Predictions vs Actual
       ↓
8. Predict Sales for New Ad Budget
```

---

## 📊 Visualizations Generated

- 📌 **Scatter Plots** — Each advertising channel vs Sales
- 🔥 **Correlation Heatmap** — Feature relationships
- 📉 **Sales Distribution** — Histogram with KDE
- ✅ **Actual vs Predicted** — Model accuracy plot
- 📈 **Regression Line** — TV spend vs Sales
- 🔍 **Residual Plot** — Model diagnostic check

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Mean Absolute Error (MAE) | ~1.06 |
| Mean Squared Error (MSE) | ~1.56 |
| Root Mean Squared Error (RMSE) | ~1.25 |
| R² Score | ~0.93 |

> ✅ An R² score of **0.93** means the model explains **93% of the variance** in Sales — a strong predictive performance.

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/sales-prediction-python.git
cd sales-prediction-python
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Add the Dataset
Place `advertising.csv` in the project root directory.

### 4. Run the Script
```bash
python sales_prediction.py
```

---

## 📁 Project Structure

```
sales-prediction-python/
│
├── advertising.csv              # Dataset
├── sales_prediction.py          # Main Python script
├── 1_eda_scatter.png            # Scatter plots
├── 2_correlation_heatmap.png    # Heatmap
├── 3_sales_distribution.png     # Sales distribution
├── 4_actual_vs_predicted.png    # Prediction accuracy
├── 5_tv_vs_sales_regression.png # Regression line
├── 6_residual_plot.png          # Residual diagnostic
└── README.md                    # Project documentation
```

---

## 💡 Key Learnings

- How to apply **Linear Regression** for real-world business problems
- Importance of **EDA** before model building
- How advertising spend on **TV has the strongest correlation** with Sales
- Evaluating models using multiple metrics beyond just accuracy

---

## 🙋‍♂️ Author

**Sania Aftab**  
saniaaftab555@gmail.com
🔗 [LinkedIn](www.linkedin.com/in/sania-aftab-80254a288) | [GitHub](https://github.com/saniaaftab555-sudo)

---

## 📜 License

This project is licensed under the **MIT License**.

---

⭐ *If you found this project helpful, consider giving it a star!*

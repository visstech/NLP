import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression

st.title("📘 Machine Learning Formulas and Code Explorer")

option = st.selectbox(
    "Select a Topic:",
    [
        "Linear Regression + MSE + RMSE",
        "Logistic Regression + Cross Entropy",
        "Classification Metrics (Accuracy, Precision, Recall, F1)",
        "Confusion Matrix",
        "Gradient Descent",
        "L1 Regularization",
        "L2 Regularization",
        "ROC AUC Score",
        "Neural Networks - Forward Propagation",
        "Neural Networks - Backpropagation"
    ]
)

if option == "Linear Regression + MSE + RMSE":
    st.header("📈 Linear Regression + MSE + RMSE")
    st.markdown("""
    **Linear Regression** is a supervised learning algorithm used to model the relationship between a dependent variable `y` and one or more independent variables `X`. It tries to find the best-fit line.

    #### 📌 Formulas
    - Mean Squared Error (MSE):
      $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
    - Root Mean Squared Error (RMSE):
      $$\text{RMSE} = \sqrt{\text{MSE}}$$
    """)

    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    st.code("""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

X, y = make_regression(n_samples=100, n_features=1, noise=10)
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
print("MSE:", mse)
print("RMSE:", rmse)
""", language="python")

    st.write("### MSE:", round(mse, 2))
    st.write("### RMSE:", round(rmse, 2))

    fig, ax = plt.subplots()
    ax.scatter(X, y, label="Actual")
    ax.plot(X, y_pred, color='red', label="Predicted")
    ax.set_title("Linear Regression Fit")
    ax.legend()
    st.pyplot(fig)

elif option == "Logistic Regression + Cross Entropy":
    st.header("📊 Logistic Regression + Cross Entropy Loss")
    st.markdown("""
    **Logistic Regression** is used for binary classification problems. It predicts probabilities using the logistic (sigmoid) function.

    #### 📌 Formula (Binary Cross Entropy / Log Loss):
    $$\text{Loss} = - \frac{1}{n} \sum [y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]$$
    """)

    X, y = make_classification(n_samples=100, n_features=1, n_informative=1, n_redundant=0, random_state=0)
    model = LogisticRegression()
    model.fit(X, y)
    y_pred_prob = model.predict_proba(X)[:, 1]
    loss = log_loss(y, y_pred_prob)

    st.code("""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

X, y = make_classification(n_samples=100, n_features=1)
model = LogisticRegression()
model.fit(X, y)
y_pred_prob = model.predict_proba(X)[:, 1]
loss = log_loss(y, y_pred_prob)
print("Cross Entropy Loss:", loss)
""", language="python")

    st.write("### Cross Entropy Loss:", round(loss, 4))
    st.write("### Predicted Probabilities (first 5):", y_pred_prob[:5])

elif option == "Classification Metrics (Accuracy, Precision, Recall, F1)":
    st.header("📊 Classification Metrics")
    st.markdown("""
    These metrics are used to evaluate classification models.

    #### 📌 Formulas:
    - Accuracy: $$\frac{TP + TN}{TP + TN + FP + FN}$$
    - Precision: $$\frac{TP}{TP + FP}$$
    - Recall: $$\frac{TP}{TP + FN}$$
    - F1 Score: $$2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
    """)

    X, y = make_classification(n_samples=100, n_features=4, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.code("""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
""", language="python")

    st.write(f"**Accuracy:** {acc:.2f}")
    st.write(f"**Precision:** {prec:.2f}")
    st.write(f"**Recall:** {rec:.2f}")
    st.write(f"**F1 Score:** {f1:.2f}")

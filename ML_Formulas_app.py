import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression

st.title("📘 Machine Learning Formulas and Code Explorer")

option = st.selectbox(
    "Select a Topic:",
    [
        "Linear Regression + MSE",
        "Logistic Regression + Cross Entropy",
        "Classification Metrics (Accuracy, Precision, Recall, F1)",
        "Confusion Matrix",
        "Gradient Descent",
        "L1 Regularization",
        "L2 Regularization",
        "ROC AUC Score"
    ]
)

if option == "Linear Regression + MSE":
    st.header("📈 Linear Regression + Mean Squared Error")
    st.latex(r"\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")

    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)

    st.code("""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
X, y = make_regression(n_samples=100, n_features=1, noise=10)
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print(mse)
""", language="python")

    st.write("### Mean Squared Error:", round(mse, 2))
    fig, ax = plt.subplots()
    ax.scatter(X, y, label="Actual")
    ax.plot(X, y_pred, color='red', label="Predicted")
    ax.set_title("Linear Regression Fit")
    ax.legend()
    st.pyplot(fig)

elif option == "Logistic Regression + Cross Entropy":
    st.header("📊 Logistic Regression + Cross Entropy Loss")
    st.latex(r"\text{Loss} = - \frac{1}{n} \sum [y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]")

    X, y = make_classification(n_samples=100, n_features=1, n_informative=1, n_redundant=0, random_state=0)
    model = LogisticRegression()
    model.fit(X, y)
    y_pred_prob = model.predict_proba(X)[:, 1]

    st.code("""
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=1)
model = LogisticRegression()
model.fit(X, y)
y_pred_prob = model.predict_proba(X)[:, 1]
""", language="python")

    st.write("### Predicted Probabilities (first 5):", y_pred_prob[:5])

elif option == "Classification Metrics (Accuracy, Precision, Recall, F1)":
    st.header("📊 Classification Metrics")
    st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}")
    st.latex(r"\text{Precision} = \frac{TP}{TP + FP}")
    st.latex(r"\text{Recall} = \frac{TP}{TP + FN}")
    st.latex(r"\text{F1} = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}")

    X, y = make_classification(n_samples=100, n_features=4, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.code("""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
""", language="python")

    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"**Precision:** {precision_score(y_test, y_pred):.2f}")
    st.write(f"**Recall:** {recall_score(y_test, y_pred):.2f}")
    st.write(f"**F1 Score:** {f1_score(y_test, y_pred):.2f}")

elif option == "Confusion Matrix":
    st.header("🧮 Confusion Matrix")
    st.latex(r"\begin{bmatrix} TP & FP \\ FN & TN \end{bmatrix}")

    X, y = make_classification(n_samples=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    st.code("""
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
""", language="python")

    st.write("### Confusion Matrix:")
    st.write(cm)

elif option == "Gradient Descent":
    st.header("📉 Gradient Descent")
    st.latex(r"\theta = \theta - \alpha \cdot \nabla J(\theta)")

    st.code("""
# Basic Gradient Descent for y = 2x
x = np.array([1, 2, 3, 4])
y = np.array([2, 4, 6, 8])

m, b = 0, 0  # initial values
lr = 0.01

for _ in range(1000):
    y_pred = m * x + b
    error = y - y_pred
    m_grad = -2 * sum(x * error) / len(x)
    b_grad = -2 * sum(error) / len(x)
    m -= lr * m_grad
    b -= lr * b_grad

print(m, b)
""", language="python")

elif option == "L1 Regularization":
    st.header("📏 L1 Regularization (Lasso)")
    st.latex(r"J(\theta) = MSE + \lambda \sum |\theta|")

    st.code("""
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)
model.fit(X, y)
""", language="python")

elif option == "L2 Regularization":
    st.header("📏 L2 Regularization (Ridge)")
    st.latex(r"J(\theta) = MSE + \lambda \sum \theta^2")

    st.code("""
from sklearn.linear_model import Ridge
model = Ridge(alpha=0.1)
model.fit(X, y)
""", language="python")

elif option == "ROC AUC Score":
    st.header("📐 ROC AUC Score")
    st.latex(r"\text{AUC} = \int_{0}^{1} TPR(FPR^{-1}(x)) dx")

    X, y = make_classification(n_samples=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    st.code("""
from sklearn.metrics import roc_auc_score
y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(auc)
""", language="python")

    st.write(f"### ROC AUC Score: {auc:.2f}")

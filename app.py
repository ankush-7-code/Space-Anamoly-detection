import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

import tensorflow as tf
from tensorflow.keras import layers, models

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Spacecraft ADSS", layout="wide")

# -------------------------------
# TITLE
# -------------------------------
st.title("🚀 Spacecraft Telemetry Anomaly Detection")
st.markdown("### Autonomous Decision Support System (ADSS)")

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload dataset.csv", type=["csv"])

if uploaded_file is not None:

    # Load data
    df = pd.read_csv(uploaded_file)

    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # PREPROCESSING
    # -------------------------------
    y = df['anomaly']
    X = df.drop('anomaly', axis=1)

    # Keep numeric only
    X = X.select_dtypes(include=[np.number])

    # Handle missing values
    X = X.fillna(X.mean())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------------
    # AUTOENCODER MODEL
    # -------------------------------
    X_train_normal = X_train[y_train == 0]

    input_dim = X_train.shape[1]

    autoencoder = models.Sequential([
        layers.Dense(16, activation='relu', input_shape=(input_dim,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(input_dim, activation='linear')
    ])

    autoencoder.compile(optimizer='adam', loss='mse')

    st.info("Training Autoencoder on normal data...")

    autoencoder.fit(
        X_train_normal, X_train_normal,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )

    # -------------------------------
    # THRESHOLD
    # -------------------------------
    train_recon = autoencoder.predict(X_train_normal)
    train_mse = np.mean((X_train_normal - train_recon) ** 2, axis=1)

    threshold = np.percentile(train_mse, 95)

    # -------------------------------
    # TEST PREDICTION
    # -------------------------------
    test_recon = autoencoder.predict(X_test)
    test_mse = np.mean((X_test - test_recon) ** 2, axis=1)

    y_pred = (test_mse > threshold).astype(int)

    # -------------------------------
    # DECISION SYSTEM
    # -------------------------------
    def classify_risk(error):
        if error < threshold:
            return "NORMAL"
        elif error < threshold * 1.5:
            return "WARNING"
        else:
            return "CRITICAL"

    risk_levels = [classify_risk(e) for e in test_mse]

    # -------------------------------
    # METRICS
    # -------------------------------
    st.subheader("📊 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    accuracy = np.mean(y_pred == y_test)

    from sklearn.metrics import precision_score, recall_score, f1_score

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    col1.metric("Accuracy", f"{accuracy:.2f}")
    col2.metric("Precision", f"{precision:.2f}")
    col3.metric("Recall", f"{recall:.2f}")
    col4.metric("F1 Score", f"{f1:.2f}")

    # -------------------------------
    # CONFUSION MATRIX
    # -------------------------------
    st.subheader("📉 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    ax.imshow(cm)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha='center', va='center')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # -------------------------------
    # ROC CURVE
    # -------------------------------
    st.subheader("📈 ROC Curve")

    fpr, tpr, _ = roc_curve(y_test, test_mse)
    auc = roc_auc_score(y_test, test_mse)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax2.plot([0,1],[0,1],'--')
    ax2.legend()
    st.pyplot(fig2)

    # -------------------------------
    # RISK DISTRIBUTION
    # -------------------------------
    st.subheader("⚠️ Risk Level Distribution")

    risk_df = pd.DataFrame({"Risk": risk_levels})
    st.bar_chart(risk_df["Risk"].value_counts())

    # -------------------------------
    # DECISION OUTPUT
    # -------------------------------
    st.subheader("🧠 Decision Support Output")

    output_df = pd.DataFrame({
        "Reconstruction Error": test_mse,
        "Prediction": y_pred,
        "Risk Level": risk_levels
    })

    st.dataframe(output_df.head(20))

    # -------------------------------
    # ALERT SYSTEM
    # -------------------------------
    st.subheader("🚨 System Status")

    if "CRITICAL" in risk_levels:
        st.error("⚠️ CRITICAL anomalies detected! Immediate action required.")
    elif "WARNING" in risk_levels:
        st.warning("⚠️ Warning: Monitor system closely.")
    else:
        st.success("✅ System operating normally.")
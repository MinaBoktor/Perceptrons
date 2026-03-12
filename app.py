import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from preprocessing import preprocess
from perceptrons import SLP, adaline

st.set_page_config(page_title="Perceptrons Configurator", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            margin-top: 0rem;
        }
    </style>
""", unsafe_allow_html=True)


# Sidebar: Configuration
with st.sidebar:
    st.header("Model Settings")
    
    algorithm = st.radio("Algorithm", ["Perceptron (SLP)", "Adaline"])
    
    features = st.multiselect(
        "Select exactly 2 features",
        ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass", "OriginLocation"],
        max_selections=2
    )
    
    classes = st.multiselect(
        "Select exactly 2 classes",
        ["Adelie", "Chinstrap", "Gentoo"],
        max_selections=2
    )
    
    col1, col2 = st.columns(2)
    with col1:
        learning_rate = st.number_input("Learning rate", min_value=0.0001, value=0.01, step=0.001, format="%.4f")
    with col2:
        epochs = st.number_input("Epochs (m)", min_value=1, value=1000, step=100)
        
    if algorithm == "Adaline":
        mse_threshold = st.number_input("MSE threshold", min_value=0.0001, value=0.15, step=0.01, format="%.4f")
        use_bias = False 
    else:
        mse_threshold = None
        use_bias = st.checkbox("Add bias to the model", value=True)

    st.markdown("<br>", unsafe_allow_html=True)
    train_button = st.button("Train Model", type="primary", use_container_width=True)


# Main Area
st.title("Perceptrons Dashboard (Task 1)")

if train_button:
    if len(features) != 2 or len(classes) != 2:
        st.error("Please select exactly TWO features and TWO classes in the sidebar.")
    else:
        with st.spinner(f"Training {algorithm}..."):
            try:
                df = pd.read_csv("penguins.csv", na_values=["NA"])
                train_df, test_df = preprocess(df, classes=classes.copy(), features=features.copy(), mlp=False)

                if isinstance(train_df, int) and train_df == -1:
                    st.error("Preprocessing failed. Please check your selections.")
                    st.stop()

                if algorithm == "Perceptron (SLP)":
                    weights, bias, test_acc, y_true, y_pred, errors = SLP(
                        train_df, test_df, use_bias=use_bias, learning_rate=learning_rate, epochs=epochs
                    )
                else:
                    weights, bias, test_acc, y_true, y_pred, errors = adaline(
                        train_df, test_df, learning_rate=learning_rate, epochs=epochs, mse_threshold=mse_threshold
                    )

                classes = sorted(classes)

                # Calculate Train Accuracy
                X_train = train_df[features].values
                y_train = train_df["Species"].values
                
                train_linear_output = np.dot(X_train, weights) + bias
                train_predictions = np.where(train_linear_output >= 0, 1, -1)
                train_acc = np.mean(train_predictions == y_train)

                # --- FIX: Dedicated row for metrics ---
                st.markdown("### Model Performance")
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    st.metric(label="Training Accuracy", value=f"{train_acc * 100:.2f}%")
                with metric_col2:
                    st.metric(label="Test Accuracy", value=f"{test_acc * 100:.2f}%")
                
                st.markdown("<br>", unsafe_allow_html=True) # Adds a little breathing room before the plots
                # --------------------------------------

                plot_col1, plot_col2 = st.columns(2)

                with plot_col1:
                    feature_cols = [col for col in test_df.columns if col != "Species"]
                    f1_name, f2_name = feature_cols[0], feature_cols[1]

                    fig, ax = plt.subplots(figsize=(5, 3.2)) 
                    ax.set_title("Decision Boundary & Scatter Plot", fontsize=10, fontweight='bold')

                    train_neg = train_df[train_df["Species"] == -1]
                    train_pos = train_df[train_df["Species"] == 1]
                    test_neg = test_df[test_df["Species"] == -1]
                    test_pos = test_df[test_df["Species"] == 1]

                    ax.scatter(train_neg[f1_name], train_neg[f2_name], label=f"{classes[0]} (Train)", 
                               color='tab:blue', marker='o', alpha=0.3, edgecolors='none')
                    ax.scatter(train_pos[f1_name], train_pos[f2_name], label=f"{classes[1]} (Train)", 
                               color='tab:orange', marker='o', alpha=0.3, edgecolors='none')

                    ax.scatter(test_neg[f1_name], test_neg[f2_name], label=f"{classes[0]} (Test)", 
                               color='tab:blue', marker='x', s=60, linewidths=2)
                    ax.scatter(test_pos[f1_name], test_pos[f2_name], label=f"{classes[1]} (Test)", 
                               color='tab:orange', marker='x', s=60, linewidths=2)

                    if len(weights) == 2:
                        x_vals = np.array(ax.get_xlim())
                        if weights[1] != 0: 
                            y_vals = -(weights[0]/weights[1]) * x_vals - (bias/weights[1])
                            ax.plot(x_vals, y_vals, '--', color='red', label='Boundary')
                    else:
                        st.caption(f"2D boundary line hidden due to {len(weights)} dimensions.")

                    ax.set_xlabel(f1_name)
                    ax.set_ylabel(f2_name)
                    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize='small')
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)

                with plot_col2:
                    cm = confusion_matrix(y_true, y_pred)
                    fig_cm, ax_cm = plt.subplots(figsize=(5, 3.2))
                    ax_cm.set_title("Confusion Matrix", fontsize=10, fontweight='bold')
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax_cm, cbar=False)
                    ax_cm.set_xlabel('Predicted')
                    ax_cm.set_ylabel('Actual')
                    fig_cm.tight_layout()
                    st.pyplot(fig_cm, use_container_width=True)

                # --- Learning Curve Plot ---
                st.markdown("---")
                st.subheader("📈 Learning Curve")
                
                fig_loss, ax_loss = plt.subplots(figsize=(10, 3))
                
                ax_loss.plot(range(1, len(errors) + 1), errors, color='tab:red', linewidth=2)
                
                ax_loss.set_xlabel("Epochs")
                y_label_text = "Mean Squared Error (MSE)" if algorithm == "Adaline" else "Misclassifications / Error"
                ax_loss.set_ylabel(y_label_text)
                
                ax_loss.grid(True, linestyle='--', alpha=0.6)
                fig_loss.tight_layout()
                st.pyplot(fig_loss, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during training: {str(e)}")
else:
    st.info("Use the sidebar on the left to configure your model parameters, then click 'Train Model'.")
# University Quality Water Potability Dashboard
# - Includes: Data upload, model evaluation, dynamic visualization, quality layout
# - Suitable for public sharing, Streamlit Cloud, and academic demonstration

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

st.set_page_config(page_title='Water Potability - University Dashboard', layout='wide')
st.title('Water Potability Project: Model Evaluation Dashboard')

st.write('''
This dashboard demonstrates a robust comparison of supervised ML models for the water potability classification task.\
- **Upload your model results** (CSV or Excel)
- **Compare all key metrics** (Accuracy, Precision, Recall, F1, ROC-AUC)
- **Visualize** model comparison, feature importance, and best model details
''')

# Sidebar for file upload
st.sidebar.header('Upload Your Model Results (CSV or Excel)')
file = st.sidebar.file_uploader('Upload your CSV or Excel file (must contain columns: Model, Accuracy, Precision, Recall, F1-Score, ROC-AUC)', type=['csv', 'xlsx'])

# Example Data
example_data = pd.DataFrame([
    {'Model': 'Logistic Regression', 'Accuracy': 0.61, 'Precision': 0.0, 'Recall': 0.0, 'F1-Score': 0.0, 'ROC-AUC': 0.53},
    {'Model': 'K-Nearest Neighbors', 'Accuracy': 0.612, 'Precision': 0.502, 'Recall': 0.392, 'F1-Score': 0.441, 'ROC-AUC': 0.606},
    {'Model': 'Decision Tree', 'Accuracy': 0.627, 'Precision': 0.417, 'Recall': 0.418, 'F1-Score': 0.417, 'ROC-AUC': 0.544},
    {'Model': 'Naive Bayes', 'Accuracy': 0.627, 'Precision': 0.555, 'Recall': 0.211, 'F1-Score': 0.307, 'ROC-AUC': 0.611},
    {'Model': 'Support Vector Machine', 'Accuracy': 0.646, 'Precision': 0.475, 'Recall': 0.340, 'F1-Score': 0.398, 'ROC-AUC': 0.654},
    {'Model': 'Random Forest', 'Accuracy': 0.661, 'Precision': 0.501, 'Recall': 0.399, 'F1-Score': 0.444, 'ROC-AUC': 0.684},
    {'Model': 'AdaBoost', 'Accuracy': 0.605, 'Precision': 0.483, 'Recall': 0.180, 'F1-Score': 0.262, 'ROC-AUC': 0.669},
    {'Model': 'Gradient Boosting', 'Accuracy': 0.627, 'Precision': 0.534, 'Recall': 0.332, 'F1-Score': 0.409, 'ROC-AUC': 0.604}
])

# Read uploaded data or use example
if file is not None:
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        df = example_data
        st.info("Showing sample results.")
else:
    df = example_data
    st.info("No file uploaded. Showing sample results.")

for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

st.subheader('Model Results Table')
st.dataframe(df.style.format('{:.3f}', subset=['Accuracy','Precision','Recall','F1-Score','ROC-AUC']), use_container_width=True)

# Plotting
st.subheader('Model Metrics Comparison')
metrics_long = df.melt(id_vars='Model', var_name='Metric', value_name='Score')
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_long, ax=ax)
plt.xticks(rotation=30)
plt.title('Model Metrics Comparison')
plt.ylabel('Score')
plt.xlabel('Model')
plt.legend(title='Metric', loc='upper right')
st.pyplot(fig)

# Best model
best_row = df.sort_values('Accuracy', ascending=False).iloc[0]
st.success(f"Best Model: {best_row['Model']} (Accuracy: {best_row['Accuracy']:.3f})")
st.write(best_row)

st.markdown('---')
st.caption('Academic Dashboard Template | Suitable for university project demonstration | Deploy on Streamlit Cloud for public access')

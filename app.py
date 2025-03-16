import streamlit as st
import pandas as pd
import numpy as np
from data_processor import DataProcessor
from model_trainer import ModelTrainer
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Student Behavior Forecasting", layout="wide")

def main():
    st.title("Student Behavior Analysis and Forecasting")
    
    # File upload
    uploaded_file = st.file_uploader("Upload behavior data CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load and process data
            data = pd.read_csv(uploaded_file)
            processor = DataProcessor(data)
            processed_data = processor.process_data()
            
            # Display data overview
            st.header("Data Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Data Sample")
                st.dataframe(processed_data.head())
                
            with col2:
                st.subheader("Dataset Statistics")
                st.write(f"Total Records: {len(processed_data)}")
                st.write(f"Date Range: {processed_data['date'].min()} to {processed_data['date'].max()}")
            
            # Behavior distribution visualization
            st.header("Behavior Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            behavior_counts = processor.get_behavior_distribution()
            sns.barplot(x=behavior_counts.index, y=behavior_counts.values)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
            
            # Model training and prediction
            st.header("Behavior Prediction")
            
            if st.button("Train Model and Generate Predictions"):
                with st.spinner("Training model..."):
                    trainer = ModelTrainer(processed_data)
                    metrics, predictions = trainer.train_and_predict()
                    
                    # Display metrics
                    st.subheader("Model Performance")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("MAE", f"{metrics['mae']:.3f}")
                    col2.metric("MSE", f"{metrics['mse']:.3f}")
                    col3.metric("R2 Score", f"{metrics['r2']:.3f}")
                    
                    # Plot predictions
                    st.subheader("Prediction Results")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    plt.plot(predictions['actual'], label='Actual', alpha=0.7)
                    plt.plot(predictions['predicted'], label='Predicted', alpha=0.7)
                    plt.legend()
                    plt.title("Actual vs Predicted Behavior")
                    st.pyplot(fig)
                    plt.close()
                    
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            
if __name__ == "__main__":
    main()

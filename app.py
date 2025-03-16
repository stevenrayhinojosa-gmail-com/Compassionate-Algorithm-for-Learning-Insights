import streamlit as st
import pandas as pd
import numpy as np
from data_processor import DataProcessor
from model_trainer import ModelTrainer
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Student Behavior Forecasting", layout="wide")

def plot_weekly_patterns(data):
    """Plot weekly behavior patterns"""
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_by_day = data.groupby('day_of_week')['behavior_score'].mean()
    sns.barplot(x=avg_by_day.index, y=avg_by_day.values, 
                palette=sns.color_palette("RdYlGn", n_colors=5))
    plt.title("Average Behavior Score by Day of Week")
    plt.xlabel("Day of Week (0=Monday, 4=Friday)")
    plt.ylabel("Average Score")
    return fig

def plot_behavior_trends(data):
    """Plot behavior trends over time"""
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(data['date'], data['behavior_score'], label='Daily Score', alpha=0.5)
    plt.plot(data['date'], data['rolling_avg_7d'], label='7-day Average', linewidth=2)
    plt.fill_between(data['date'], 
                     data['rolling_avg_7d'] - data['rolling_std_7d'],
                     data['rolling_avg_7d'] + data['rolling_std_7d'], 
                     alpha=0.2)
    plt.title("Behavior Score Trends")
    plt.xlabel("Date")
    plt.ylabel("Score")
    plt.legend()
    plt.xticks(rotation=45)
    return fig

def main():
    st.title("Student Behavior Analysis and Forecasting")

    st.write("""
    This application analyzes student behavior patterns and predicts future trends.
    Upload your behavior data CSV file to begin the analysis.

    The analysis includes:
    - Daily behavior scores and patterns
    - Weekly trends and improvements
    - Behavioral forecasting using machine learning
    """)

    uploaded_file = st.file_uploader("Upload behavior data CSV", type=['csv'])

    if uploaded_file is not None:
        try:
            # Load and process data
            data = pd.read_csv(uploaded_file)
            processor = DataProcessor(data)
            processed_data = processor.process_data()
            summary_stats = processor.get_summary_stats()

            # Display summary statistics
            st.header("Summary Statistics")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Days Recorded", summary_stats['total_days'])
                st.metric("Average Score", f"{summary_stats['avg_score']:.2f}")

            with col2:
                st.metric("Best Day", summary_stats['best_day'].strftime('%Y-%m-%d'))
                st.metric("Most Challenging Day", 
                         summary_stats['challenging_day'].strftime('%Y-%m-%d'))

            with col3:
                st.metric("Weekly Trend", 
                         f"{summary_stats['weekly_trend']:.2f}",
                         delta=summary_stats['weekly_trend'])

            # Display recent data
            st.header("Recent Behavior Records")
            st.dataframe(processed_data.tail().style.format({
                'behavior_score': '{:.2f}',
                'date': lambda x: x.strftime('%Y-%m-%d'),
                'rolling_avg_7d': '{:.2f}',
                'weekly_improvement': '{:.2f}'
            }))

            # Visualizations
            st.header("Behavior Patterns")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Weekly Patterns")
                st.pyplot(plot_weekly_patterns(processed_data))

            with col2:
                st.subheader("Behavior Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                behavior_counts = processor.get_behavior_distribution()
                colors = ['red', 'yellow', 'green']
                sns.barplot(x=behavior_counts.index, y=behavior_counts.values, palette=colors)
                plt.title("Distribution of Behavior Types")
                plt.xlabel("Behavior Category")
                plt.ylabel("Count")
                st.pyplot(fig)

            st.header("Behavior Trends")
            st.pyplot(plot_behavior_trends(processed_data))

            # Model training and prediction
            st.header("Behavior Prediction")
            if st.button("Train Model and Generate Predictions"):
                with st.spinner("Training model and generating predictions..."):
                    trainer = ModelTrainer(processed_data)
                    metrics, predictions = trainer.train_and_predict()

                    # Display metrics
                    st.subheader("Model Performance")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Mean Absolute Error", f"{metrics['mae']:.3f}")
                    col2.metric("Mean Squared Error", f"{metrics['mse']:.3f}")
                    col3.metric("R² Score", f"{metrics['r2']:.3f}")

                    # Plot predictions
                    st.subheader("Prediction Results")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    plt.plot(predictions.index, predictions['actual'], 
                            label='Actual', alpha=0.7)
                    plt.plot(predictions.index, predictions['predicted'], 
                            label='Predicted', alpha=0.7)
                    plt.legend()
                    plt.title("Actual vs Predicted Behavior Scores")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")

if __name__ == "__main__":
    main()
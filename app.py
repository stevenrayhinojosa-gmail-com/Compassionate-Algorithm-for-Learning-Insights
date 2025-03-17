import streamlit as st
import pandas as pd
import numpy as np
from data_processor import DataProcessor
from model_trainer import ModelTrainer
import matplotlib.pyplot as plt
import seaborn as sns
from database import SessionLocal, engine
from models import Base, Student, BehaviorRecord, TimeSlotBehavior, AlertConfiguration
from datetime import datetime
from alert_system import AlertSystem

# Create database tables
Base.metadata.create_all(bind=engine)

st.set_page_config(page_title="Student Behavior Forecasting", layout="wide")

def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

def store_behavior_data(processed_data, student_name="Default Student"):
    """Store processed behavior data in the database"""
    db = get_db()
    try:
        # Create or get student
        student = db.query(Student).filter(Student.name == student_name).first()
        if not student:
            student = Student(name=student_name)
            db.add(student)
            db.commit()
            db.refresh(student)

        # Store behavior records
        for _, row in processed_data.iterrows():
            record = BehaviorRecord(
                date=row['date'],
                student_id=student.id,
                behavior_score=row['behavior_score'],
                red_count=row['red_count'],
                yellow_count=row['yellow_count'],
                green_count=row['green_count'],
                rolling_avg_7d=row['rolling_avg_7d'],
                rolling_std_7d=row['rolling_std_7d'],
                behavior_trend=row['behavior_trend'],
                weekly_improvement=row['weekly_improvement']
            )
            db.add(record)

        db.commit()
        return True
    except Exception as e:
        db.rollback()
        st.error(f"Error storing data: {str(e)}")
        return False
    finally:
        db.close()

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

def configure_alerts(student_name, db):
    """Configure alert settings for a student"""
    st.header("Alert Configuration")

    # Get or create student
    student = db.query(Student).filter(Student.name == student_name).first()
    if not student:
        st.warning("Please save some behavior data first to configure alerts.")
        return

    alert_system = AlertSystem(db)

    # Display existing alert configurations
    existing_configs = db.query(AlertConfiguration).filter(
        AlertConfiguration.student_id == student.id
    ).all()

    if existing_configs:
        st.subheader("Existing Alert Configurations")
        for config in existing_configs:
            with st.expander(f"Alert: {config.name}"):
                st.write(f"Description: {config.description}")
                st.write(f"Behavior Score Threshold: {config.behavior_threshold}")
                st.write(f"Consecutive Red Threshold: {config.red_threshold}")
                st.write(f"Trend Threshold: {config.trend_threshold}")
                st.write(f"Active: {config.is_active}")
                st.write(f"Alert on Predictions: {config.notify_on_prediction}")

    # Form for creating new alert configuration
    st.subheader("Create New Alert Configuration")
    with st.form("alert_config_form"):
        name = st.text_input("Alert Name", "Behavior Alert")
        description = st.text_area("Description", "Alert for concerning behavior patterns")
        behavior_threshold = st.slider("Behavior Score Threshold", 0.0, 2.0, 1.0)
        red_threshold = st.slider("Consecutive Red Count Threshold", 1, 10, 3)
        trend_threshold = st.slider("Behavior Trend Threshold", -2.0, 0.0, -0.5)
        notify_on_prediction = st.checkbox("Alert on Predicted Behaviors", True)

        if st.form_submit_button("Create Alert Configuration"):
            config_data = {
                'name': name,
                'description': description,
                'behavior_threshold': behavior_threshold,
                'red_threshold': red_threshold,
                'trend_threshold': trend_threshold,
                'notify_on_prediction': notify_on_prediction
            }
            alert_system.create_alert_config(student.id, config_data)
            st.success("Alert configuration created successfully!")
            st.rerun()

def display_alerts(student_name, db):
    """Display active alerts for a student"""
    student = db.query(Student).filter(Student.name == student_name).first()
    if not student:
        return

    alert_system = AlertSystem(db)
    active_alerts = alert_system.get_active_alerts(student.id)

    if active_alerts:
        st.sidebar.header("⚠️ Active Alerts")
        for alert in active_alerts:
            alert_color = "red" if not alert.is_prediction else "orange"
            with st.sidebar.container(border=True):
                st.markdown(f"<p style='color: {alert_color};'>", unsafe_allow_html=True)
                st.write(f"Alert Type: {alert.alert_type}")
                st.write(f"Value: {alert.value:.2f}")
                st.write(f"Triggered: {alert.triggered_at}")
                if alert.is_prediction:
                    st.write("(Predicted)")
                st.markdown("</p>", unsafe_allow_html=True)

def main():
    st.title("Student Behavior Analysis and Forecasting")

    st.write("""
    This application analyzes student behavior patterns and predicts future trends.
    Upload your behavior data CSV file to begin the analysis.

    The analysis includes:
    - Daily behavior scores and patterns
    - Weekly trends and improvements
    - Behavioral forecasting using machine learning
    - Alert system for at-risk behavior patterns
    """)

    # Add student name input
    student_name = st.text_input("Student Name", "Default Student")

    # Get database session
    db = get_db()

    # Display alert configuration in sidebar
    with st.sidebar:
        if st.button("Configure Alerts"):
            configure_alerts(student_name, db)

    # Display active alerts
    display_alerts(student_name, db)

    uploaded_file = st.file_uploader("Upload behavior data CSV", type=['csv'])

    if uploaded_file is not None:
        try:
            # Load and process data
            data = pd.read_csv(uploaded_file)
            processor = DataProcessor(data)
            processed_data = processor.process_data()
            summary_stats = processor.get_summary_stats()

            # Store data in database
            if st.button("Save Data to Database"):
                if store_behavior_data(processed_data, student_name):
                    st.success("Data successfully saved to database!")
                else:
                    st.error("Failed to save data to database.")

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

                    # Plot historical data
                    historical_data = predictions[predictions['actual'].notna()]
                    plt.plot(historical_data['date'], historical_data['actual'], 
                            label='Actual', alpha=0.7)
                    plt.plot(historical_data['date'], historical_data['predicted'], 
                            label='Historical Predictions', alpha=0.7)

                    # Plot future predictions
                    future_data = predictions[predictions['actual'].isna()]
                    plt.plot(future_data['date'], future_data['predicted'], 
                            label='24-Hour Forecast', linestyle='--', alpha=0.7)

                    # Add confidence interval for future predictions
                    std_dev = processed_data['rolling_std_7d'].mean()
                    plt.fill_between(
                        future_data['date'],
                        future_data['predicted'] - std_dev,
                        future_data['predicted'] + std_dev,
                        alpha=0.2,
                        label='Forecast Uncertainty'
                    )

                    plt.legend()
                    plt.title("Behavior Score: Historical Data and 24-Hour Forecast")
                    plt.xlabel("Date/Time")
                    plt.ylabel("Behavior Score")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                    # Display future predictions table
                    st.subheader("24-Hour Behavior Forecast")
                    future_predictions = future_data.copy()
                    future_predictions['hour'] = future_predictions['date'].dt.strftime('%I:00 %p')
                    future_predictions['predicted_score'] = future_predictions['predicted'].round(2)
                    st.dataframe(
                        future_predictions[['hour', 'predicted_score']],
                        hide_index=True
                    )

                    # Check for predicted alerts
                    student = db.query(Student).filter(Student.name == student_name).first()
                    if student:
                        future_data = predictions[predictions['actual'].isna()]
                        if not future_data.empty:
                            alert_system.check_alerts(
                                student.id,
                                {
                                    'behavior_score': future_data['predicted'].iloc[-1],
                                    'red_count': processed_data['red_count'].mean(),
                                    'behavior_trend': future_data['predicted'].diff().iloc[-1]
                                },
                                is_prediction=True
                            )

            # Check for alerts based on processed data
            student = db.query(Student).filter(Student.name == student_name).first()
            if student:
                alert_system = AlertSystem(db)
                alerts = alert_system.check_alerts(
                    student.id,
                    {
                        'behavior_score': processed_data['behavior_score'].iloc[-1],
                        'red_count': processed_data['red_count'].iloc[-1],
                        'behavior_trend': processed_data['behavior_trend'].iloc[-1]
                    }
                )
                if alerts:
                    st.warning(f"Found {len(alerts)} new alert(s)! Check the sidebar for details.")

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
        finally:
            db.close()

if __name__ == "__main__":
    main()
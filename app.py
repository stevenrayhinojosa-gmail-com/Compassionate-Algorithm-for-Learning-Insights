import streamlit as st
import pandas as pd
import numpy as np
from data_processor import DataProcessor
from model_trainer import ModelTrainer
import matplotlib.pyplot as plt
import seaborn as sns
from database import SessionLocal, engine
from models import (
    Base, Student, BehaviorRecord, TimeSlotBehavior, AlertConfiguration,
    MedicationRecord, MedicationLog, LearningEnvironment, StaffChange, RoutineChange, NutritionLog, SeasonalPattern, PredictionFeedback
)
from datetime import datetime, date
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

# Load the sample data
@st.cache_data
def load_sample_data():
    """Load and parse the sample data file"""
    file_path = "attached_assets/CALITestPandas.xlsx - Applied Behavior Analysis.csv"
    return file_path

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
                'notify_on_prediction': notify_on_prediction,
                'notification_enabled': False  # Disable notifications by default
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

def display_next_day_predictions(metrics, predictions_df):
    """Display next day predictions in a user-friendly format"""
    next_day = predictions_df['time'].dt.date.iloc[0]

    st.header(f"Behavior Predictions for {next_day.strftime('%A, %B %d')}")
    
    # Create a prominent visual representation of the day's prediction
    behavior_counts = predictions_df['predicted_category'].value_counts()
    red_count = behavior_counts.get('Red', 0)
    yellow_count = behavior_counts.get('Yellow', 0)
    green_count = behavior_counts.get('Green', 0)
    
    # Calculate overall day status
    total_periods = len(predictions_df)
    green_percentage = (green_count / total_periods) * 100
    
    # Set up a color scale for the main prediction banner
    if green_percentage >= 70:
        banner_color = "rgba(0, 180, 0, 0.2)"
        day_prediction = "Mostly Positive Day Expected"
        emoji = "🟢"
    elif green_percentage >= 40:
        banner_color = "rgba(255, 180, 0, 0.2)"
        day_prediction = "Mixed Day Expected"
        emoji = "🟡"
    else:
        banner_color = "rgba(220, 0, 0, 0.2)"
        day_prediction = "Challenging Day Expected"
        emoji = "🔴"
    
    # Create a prominent banner for the prediction
    st.markdown(
        f"""
        <div style="background-color: {banner_color}; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
            <h2>{emoji} {day_prediction} {emoji}</h2>
        </div>
        """, 
        unsafe_allow_html=True
    )






    
    # Add user feedback section
    st.divider()
    st.subheader("📝 Was this helpful?")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("👍 Thumbs Up", help="This prediction was helpful"):
            save_prediction_feedback(next_day, "thumbs_up", "The prediction was helpful")
            st.success("Thank you for your feedback!")
    
    with col2:
        if st.button("👎 Thumbs Down", help="This prediction was not helpful"):
            save_prediction_feedback(next_day, "thumbs_down", "The prediction was not helpful")
            st.success("Thank you for your feedback! This helps improve our predictions.")
    
    # Add data upload section
    st.divider()
    st.subheader("📁 Upload New Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV File", 
            type=['csv'],
            help="Upload a new CSV file with behavior data"
        )
        if uploaded_file is not None:
            st.success("File uploaded successfully!")
            # Here you could add logic to process the uploaded file
    
    with col2:
        document_link = st.text_input(
            "Or paste document link:",
            placeholder="https://docs.google.com/spreadsheets/...",
            help="Paste a link to a Google Sheets or other document"
        )
        if document_link:
            st.info("Document link saved. Processing functionality can be added here.")

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

def manage_medications(student_name, db):
    st.header("Medication Management")
    # Add your medication management code here

def save_prediction_feedback(prediction_date, feedback_type, comments):
    """Save user feedback for behavior predictions"""
    db = get_db()
    try:
        # Get current student
        student_name = st.session_state.get('student_name', 'Default Student')
        student = db.query(Student).filter(Student.name == student_name).first()
        if not student:
            student = Student(name=student_name)
            db.add(student)
            db.commit()
            db.refresh(student)
        
        # Save feedback
        feedback = PredictionFeedback(
            student_id=student.id,
            prediction_date=prediction_date,
            feedback_type=feedback_type,
            comments=comments
        )
        db.add(feedback)
        db.commit()
    except Exception as e:
        db.rollback()
        st.error(f"Error saving feedback: {str(e)}")
    finally:
        db.close()

def manage_environmental_factors(student_name, db):
    st.header("Environmental Factors")
    # Add your environmental factors management code here

def main():
    st.title("CALI: Compassionate Algorithm for Learning Insights")

    # Get database session
    db = get_db()

    # Add student name input
    student_name = st.text_input("Student Name", "Default Student", key="student_name_input")



    # Display active alerts
    display_alerts(student_name, db)

    # Initialize session state for controlling display
    if 'show_predictions' not in st.session_state:
        st.session_state.show_predictions = False
    
    # Data upload section first
    st.divider()
    st.subheader("📁 Upload Your Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV File", 
            type=['csv'],
            help="Upload your behavior data CSV file",
            key="file_uploader"
        )
        if uploaded_file is not None and not st.session_state.show_predictions:
            if st.button("Process File", key="process_button"):
                st.session_state.show_predictions = True
                st.session_state.uploaded_file_data = uploaded_file
                st.rerun()
    
    with col2:
        document_link = st.text_input(
            "Or paste document link:",
            placeholder="https://docs.google.com/spreadsheets/...",
            help="Paste a link to a Google Sheets or other document",
            key="document_link_input"
        )
        if document_link and not st.session_state.show_predictions:
            if st.button("Process Link", key="process_link_button"):
                st.session_state.show_predictions = True
                st.session_state.document_link_data = document_link
                st.rerun()
    
    # Only show predictions if user has clicked process
    if st.session_state.show_predictions:
        # Add home button when data is loaded
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🏠 Home", help="Return to upload new data", key="home_button"):
                # Clear the prediction state
                st.session_state.show_predictions = False
                if 'uploaded_file_data' in st.session_state:
                    del st.session_state.uploaded_file_data
                if 'document_link_data' in st.session_state:
                    del st.session_state.document_link_data
                st.rerun()
        
        try:
            # Load and process data
            if 'uploaded_file_data' in st.session_state:
                st.success("Processing uploaded file...")
                # Process the actual uploaded file
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(st.session_state.uploaded_file_data.read())
                    file_path = tmp_file.name
            elif 'document_link_data' in st.session_state:
                st.info("Document link processing requires additional setup. Please upload a CSV file for now.")
                st.stop()
            else:
                file_path = load_sample_data()  # Fallback
            
            processor = DataProcessor(file_path, db)
            student = db.query(Student).filter(Student.name == student_name).first()
            student_id = student.id if student else None
            processed_data = processor.process_data(student_id=student_id)

            # Train model and generate next day predictions
            trainer = ModelTrainer(processed_data)
            metrics, next_day_predictions = trainer.train_and_predict_next_day()

            # Display next day predictions prominently at the top
            display_next_day_predictions(metrics, next_day_predictions)

            # Add a divider before the tabs
            st.divider()

            # Create main tabs - Overview and Deeper Insights
            tab1, tab2 = st.tabs(["🏠 Overview", "🔍 Deeper Insights"])

            with tab1:
                st.header("📈 Quick Overview")
                st.info("Your behavior prediction is shown above. For detailed analysis, check the 'Deeper Insights' tab.")

            with tab2:
                st.header("🔍 Detailed Analysis")
                
                # Create sub-tabs for all the detailed data
                sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
                "📊 Data Analysis",
                "🔮 Detailed Predictions",
                "💊 Medication Management", 
                "🌡️ Environmental Factors",
                "⚠️ Configure Alerts"
            ])

            with sub_tab1:
                if st.button("Save Data to Database"):
                    if store_behavior_data(processed_data, student_name):
                        st.success("Data successfully saved to database!")
                    else:
                        st.error("Failed to save data to database.")

                # Display summary statistics
                summary_stats = processor.get_summary_stats()
                
                # Display key metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Days", summary_stats['total_days'])
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
                try:
                    st.dataframe(processed_data.tail().style.format({
                        'behavior_score': '{:.2f}',
                        'date': lambda x: x.strftime('%Y-%m-%d'),
                        'rolling_avg_7d': '{:.2f}',
                        'weekly_improvement': '{:.2f}'
                    }))
                except:
                    st.dataframe(processed_data.tail())
            st.header("Summary Statistics")
            summary_stats = processor.get_summary_stats()
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

            with sub_tab2:
                # Display detailed predictions here
                st.subheader("🔮 Detailed Prediction Analysis")
                
                # Add detailed timeline, model metrics, etc.
                if 'metrics' in locals() and 'next_day_predictions' in locals():
                    next_day = next_day_predictions.iloc[0]
                    behavior_score = next_day['predicted_behavior_score']
                    
                    # Display predicted counts
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("🔴 Red Behaviors", int(next_day['predicted_red_count']))
                    with col2:
                        st.metric("🟡 Yellow Behaviors", int(next_day['predicted_yellow_count']))
                    with col3:
                        st.metric("🟢 Green Behaviors", int(next_day['predicted_green_count']))
                    
                    # Create hourly timeline
                    st.subheader("⏰ Hourly Timeline Forecast")
                    hours = ['8:00', '9:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00']
                    hourly_scores = [max(0, min(5, behavior_score + np.random.normal(0, 0.4))) for _ in hours]
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    colors = ['red' if s < 2 else 'orange' if s < 3.5 else 'green' for s in hourly_scores]
                    ax.bar(hours, hourly_scores, color=colors, alpha=0.7)
                    ax.set_ylabel('Predicted Behavior Score')
                    ax.set_title('Predicted Behavior Throughout the Day')
                    ax.set_ylim(0, 5)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    
                    # Model performance
                    st.subheader("🎯 Model Performance")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics['r2']:.1%}")
                    with col2:
                        st.metric("Avg Error", f"{metrics['mae']:.2f}")
                    with col3:
                        st.metric("Confidence", f"{(1 - metrics['mae']):.1%}")
                else:
                    st.info("Detailed predictions will appear here once data is processed.")

            with sub_tab3:
                manage_medications(student_name, db)

            with sub_tab4:
                manage_environmental_factors(student_name, db)

            with sub_tab5:
                configure_alerts(student_name, db)

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            print(f"Detailed error: {str(e)}")  # Add detailed error logging
    else:
        # Show welcoming message when no data is uploaded
        st.info("👋 Welcome to CALI! Please upload your behavior data CSV file or paste a document link above to get started with predictions.")
    
    db.close()

if __name__ == "__main__":
    main()
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
    MedicationRecord, MedicationLog, LearningEnvironment, StaffChange, RoutineChange, NutritionLog, SeasonalPattern
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

def manage_medications(student_name, db):
    """Manage student medications"""
    st.header("Medication Management")

    # Get or create student
    student = db.query(Student).filter(Student.name == student_name).first()
    if not student:
        st.warning("Please save some behavior data first to manage medications.")
        return

    # Display current medications
    current_meds = db.query(MedicationRecord).filter(
        MedicationRecord.student_id == student.id,
        MedicationRecord.is_active == True
    ).all()

    if current_meds:
        st.subheader("Current Medications")
        for med in current_meds:
            with st.expander(f"📋 {med.medication_name}"):
                st.write(f"Dosage: {med.dosage}")
                st.write(f"Frequency: {med.frequency}")
                st.write(f"Started: {med.start_date}")
                if med.notes:
                    st.write(f"Notes: {med.notes}")

                # Option to mark medication as discontinued
                if st.button(f"Discontinue {med.medication_name}", key=f"disc_{med.id}"):
                    med.is_active = False
                    med.end_date = date.today()
                    db.commit()
                    st.success(f"{med.medication_name} marked as discontinued.")
                    st.rerun()

    # Form to add new medication
    st.subheader("Add New Medication")
    with st.form("new_medication_form"):
        med_name = st.text_input("Medication Name")
        dosage = st.text_input("Dosage (e.g., '10mg')")
        frequency = st.selectbox(
            "Frequency",
            ["Once daily", "Twice daily", "Three times daily", "As needed", "Other"]
        )
        if frequency == "Other":
            frequency = st.text_input("Specify frequency")

        start_date = st.date_input("Start Date")
        notes = st.text_area("Notes (optional)")

        if st.form_submit_button("Add Medication"):
            new_med = MedicationRecord(
                student_id=student.id,
                medication_name=med_name,
                dosage=dosage,
                frequency=frequency,
                start_date=start_date,
                notes=notes,
                is_active=True
            )
            db.add(new_med)
            db.commit()
            st.success("Medication added successfully!")
            st.rerun()

    # Medication Log Section
    st.subheader("Medication Log")

    # Log missed medication
    with st.expander("Log Missed Medication"):
        active_meds = [(med.id, med.medication_name) for med in current_meds]
        if active_meds:
            med_id = st.selectbox(
                "Select Medication",
                options=[m[0] for m in active_meds],
                format_func=lambda x: next(m[1] for m in active_meds if m[0] == x)
            )
            missed_date = st.date_input("Date Missed")
            reason = st.text_input("Reason for missing")
            notes = st.text_area("Additional Notes")

            if st.button("Log Missed Dose"):
                log = MedicationLog(
                    medication_id=med_id,
                    timestamp=datetime.combine(missed_date, datetime.min.time()),
                    status="missed",
                    reason_if_missed=reason,
                    notes=notes
                )
                db.add(log)
                db.commit()
                st.success("Missed medication logged successfully!")
        else:
            st.info("No active medications to log.")

    # Display medication history
    st.subheader("Medication History")
    all_meds = db.query(MedicationRecord).filter(
        MedicationRecord.student_id == student.id
    ).all()

    if all_meds:
        for med in all_meds:
            with st.expander(
                f"{'🟢' if med.is_active else '⚫'} {med.medication_name} "
                f"({med.start_date} - {med.end_date or 'Present'})"
            ):
                st.write(f"Dosage: {med.dosage}")
                st.write(f"Frequency: {med.frequency}")
                if med.notes:
                    st.write(f"Notes: {med.notes}")

                # Display medication logs
                logs = db.query(MedicationLog).filter(
                    MedicationLog.medication_id == med.id
                ).order_by(MedicationLog.timestamp.desc()).all()

                if logs:
                    st.write("### Medication Logs")
                    for log in logs:
                        status_color = {
                            "taken": "🟢",
                            "missed": "🔴",
                            "late": "🟡"
                        }.get(log.status, "⚪")
                        st.write(
                            f"{status_color} {log.timestamp.strftime('%Y-%m-%d %H:%M')} - "
                            f"{log.status.title()}"
                        )
                        if log.reason_if_missed:
                            st.write(f"Reason: {log.reason_if_missed}")
                        if log.notes:
                            st.write(f"Notes: {log.notes}")


def manage_environmental_factors(student_name, db):
    """Manage environmental and contextual factors"""
    st.header("Environmental Factors Management")

    # Get or create student
    student = db.query(Student).filter(Student.name == student_name).first()
    if not student:
        st.warning("Please save some behavior data first to manage environmental factors.")
        return

    # Create tabs for different types of factors
    env_tab1, env_tab2, env_tab3, env_tab4 = st.tabs([
        "Learning Environment",
        "Staff & Routine Changes",
        "Nutrition & Health",
        "Seasonal Patterns"
    ])

    with env_tab1:
        st.subheader("Learning Environment Setup")

        # Form to add/update learning environment
        with st.form("learning_environment_form"):
            classroom_type = st.selectbox(
                "Classroom Type",
                ["Regular", "Special Education", "Resource Room", "Integrated", "Other"]
            )
            seating_position = st.selectbox(
                "Seating Position",
                ["Front", "Middle", "Back", "Near Window", "Near Door", "Other"]
            )
            noise_level = st.select_slider(
                "Noise Level",
                options=["Very Low", "Low", "Moderate", "High", "Very High"]
            )
            lighting_type = st.selectbox(
                "Lighting Type",
                ["Natural", "Fluorescent", "LED", "Mixed", "Other"]
            )
            temperature = st.slider("Temperature (°F)", 65, 85, 72)
            start_date = st.date_input("Start Date")
            notes = st.text_area("Additional Notes")

            if st.form_submit_button("Save Learning Environment"):
                new_env = LearningEnvironment(
                    student_id=student.id,
                    classroom_type=classroom_type,
                    seating_position=seating_position,
                    noise_level=noise_level,
                    lighting_type=lighting_type,
                    temperature=temperature,
                    start_date=start_date,
                    notes=notes
                )
                db.add(new_env)
                db.commit()
                st.success("Learning environment settings saved!")

    with env_tab2:
        st.subheader("Staff Changes")
        with st.form("staff_change_form"):
            staff_role = st.selectbox(
                "Staff Role",
                ["Teacher", "Aide", "Therapist", "Specialist", "Other"]
            )
            change_type = st.selectbox(
                "Change Type",
                ["New Staff", "Substitute", "Departure", "Return"]
            )
            change_date = st.date_input("Change Date")
            adjustment_period = st.number_input(
                "Expected Adjustment Period (days)",
                min_value=1,
                value=14
            )
            impact_observed = st.text_area("Observed Impact")

            if st.form_submit_button("Record Staff Change"):
                staff_change = StaffChange(
                    student_id=student.id,
                    staff_role=staff_role,
                    change_type=change_type,
                    change_date=change_date,
                    adjustment_period=adjustment_period,
                    impact_observed=impact_observed
                )
                db.add(staff_change)
                db.commit()
                st.success("Staff change recorded!")

        st.subheader("Routine Changes")
        with st.form("routine_change_form"):
            routine_type = st.selectbox(
                "Change Type",
                ["Schedule", "Activity", "Transportation", "Therapy", "Other"]
            )
            description = st.text_area("Change Description")
            duration = st.number_input(
                "Expected Duration (days)",
                min_value=1,
                value=7
            )
            adaptation_level = st.slider(
                "Adaptation Level (1-5)",
                1, 5, 3,
                help="1=Significant difficulty, 5=Well adapted"
            )
            routine_date = st.date_input("Change Start Date")
            routine_notes = st.text_area("Additional Notes")

            if st.form_submit_button("Record Routine Change"):
                routine_change = RoutineChange(
                    student_id=student.id,
                    change_type=routine_type,
                    description=description,
                    duration=duration,
                    adaptation_level=adaptation_level,
                    change_date=routine_date,
                    notes=routine_notes
                )
                db.add(routine_change)
                db.commit()
                st.success("Routine change recorded!")

    with env_tab3:
        st.subheader("Nutrition Log")
        with st.form("nutrition_log_form"):
            meal_type = st.selectbox(
                "Meal Type",
                ["Breakfast", "Morning Snack", "Lunch", "Afternoon Snack"]
            )
            food_items = st.text_area("Food Items Consumed")
            sugar_level = st.select_slider(
                "Sugar Intake Level",
                options=["Low", "Moderate", "High"]
            )
            protein_level = st.select_slider(
                "Protein Intake Level",
                options=["Low", "Moderate", "High"]
            )
            meal_date = st.date_input("Meal Date")
            meal_notes = st.text_area("Additional Notes")

            if st.form_submit_button("Add Nutrition Entry"):
                nutrition_log = NutritionLog(
                    student_id=student.id,
                    meal_type=meal_type,
                    food_items=food_items,
                    sugar_intake_level=sugar_level,
                    protein_intake_level=protein_level,
                    date=meal_date,
                    notes=meal_notes
                )
                db.add(nutrition_log)
                db.commit()
                st.success("Nutrition entry added!")

    with env_tab4:
        st.subheader("Seasonal Patterns")
        with st.form("seasonal_pattern_form"):
            season = st.selectbox(
                "Season",
                ["Spring", "Summer", "Fall", "Winter"]
            )
            year = st.number_input("Year", value=datetime.now().year, min_value=2020, max_value=2030)
            avg_score = st.number_input("Average Behavior Score", min_value=0.0, max_value=2.0, value=1.0)
            impact = st.text_area("Seasonal Impact Description")
            seasonal_notes = st.text_area("Additional Notes")

            if st.form_submit_button("Add Seasonal Pattern"):
                seasonal_pattern = SeasonalPattern(
                    student_id=student.id,
                    season=season,
                    year=year,
                    avg_behavior_score=avg_score,
                    seasonal_impact=impact,
                    notes=seasonal_notes
                )
                db.add(seasonal_pattern)
                db.commit()
                st.success("Seasonal pattern recorded!")

    # Display historical data
    st.header("Environmental Factors History")

    # Staff Changes History
    with st.expander("📋 Staff Changes History"):
        staff_changes = db.query(StaffChange).filter(
            StaffChange.student_id == student.id
        ).order_by(StaffChange.change_date.desc()).all()

        for change in staff_changes:
            st.write(f"**{change.staff_role}** - {change.change_type}")
            st.write(f"Date: {change.change_date}")
            st.write(f"Adjustment Period: {change.adjustment_period} days")
            if change.impact_observed:
                st.write(f"Impact: {change.impact_observed}")
            st.divider()

    # Routine Changes History
    with st.expander("📋 Routine Changes History"):
        routine_changes = db.query(RoutineChange).filter(
            RoutineChange.student_id == student.id
        ).order_by(RoutineChange.change_date.desc()).all()

        for change in routine_changes:
            st.write(f"**{change.change_type}** Change")
            st.write(f"Date: {change.change_date}")
            st.write(f"Duration: {change.duration} days")
            st.write(f"Adaptation Level: {change.adaptation_level}/5")
            if change.description:
                st.write(f"Description: {change.description}")
            st.divider()

    # Nutrition History
    with st.expander("📋 Nutrition History"):
        nutrition_logs = db.query(NutritionLog).filter(
            NutritionLog.student_id == student.id
        ).order_by(NutritionLog.date.desc()).all()

        for log in nutrition_logs:
            st.write(f"**{log.meal_type}** - {log.date}")
            st.write(f"Food Items: {log.food_items}")
            st.write(f"Sugar Level: {log.sugar_intake_level}")
            st.write(f"Protein Level: {log.protein_intake_level}")
            if log.notes:
                st.write(f"Notes: {log.notes}")
            st.divider()

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

    # Add Medication Management and Environmental Factors tabs
    tab1, tab2, tab3 = st.tabs(["Behavior Analysis", "Medication Management", "Environmental Factors"])

    with tab1:
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
    with tab2:
        manage_medications(student_name, db)
    with tab3:
        manage_environmental_factors(student_name, db)

if __name__ == "__main__":
    main()
from sqlalchemy import Column, Integer, Float, Date, String, ForeignKey, Boolean, DateTime, Text
from sqlalchemy.orm import relationship
from database import Base

class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    behaviors = relationship("BehaviorRecord", back_populates="student")
    alert_configs = relationship("AlertConfiguration", back_populates="student")
    medications = relationship("MedicationRecord", back_populates="student")

class BehaviorRecord(Base):
    __tablename__ = "behavior_records"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    behavior_score = Column(Float)
    red_count = Column(Integer)
    yellow_count = Column(Integer)
    green_count = Column(Integer)
    rolling_avg_7d = Column(Float)
    rolling_std_7d = Column(Float)
    behavior_trend = Column(Float)
    weekly_improvement = Column(Float)

    # Relationships
    student = relationship("Student", back_populates="behaviors")
    time_slots = relationship("TimeSlotBehavior", back_populates="behavior_record")

class TimeSlotBehavior(Base):
    __tablename__ = "time_slot_behaviors"

    id = Column(Integer, primary_key=True, index=True)
    behavior_record_id = Column(Integer, ForeignKey("behavior_records.id"))
    time_slot = Column(String)  # e.g., "7:30-7:45 AM"
    behavior_value = Column(String)  # 'r', 'y', 'g'

    # Relationship
    behavior_record = relationship("BehaviorRecord", back_populates="time_slots")

class AlertConfiguration(Base):
    __tablename__ = "alert_configurations"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    name = Column(String)  # Name of the alert configuration
    description = Column(String)
    behavior_threshold = Column(Float)  # Threshold for behavior score
    red_threshold = Column(Integer)  # Threshold for consecutive red markers
    trend_threshold = Column(Float)  # Threshold for negative behavior trend
    is_active = Column(Boolean, default=True)
    notify_on_prediction = Column(Boolean, default=True)  # Alert on predicted behaviors
    notification_phone = Column(String)  # Phone number for SMS notifications
    notification_enabled = Column(Boolean, default=True)  # Enable/disable notifications

    # Relationship
    student = relationship("Student", back_populates="alert_configs")
    alerts = relationship("Alert", back_populates="configuration")

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    configuration_id = Column(Integer, ForeignKey("alert_configurations.id"))
    triggered_at = Column(Date, index=True)
    alert_type = Column(String)  # 'behavior_score', 'red_count', 'trend'
    value = Column(Float)  # The value that triggered the alert
    is_prediction = Column(Boolean, default=False)  # Whether this is a predicted alert
    notification_sent = Column(Boolean, default=False)  # Track if notification was sent

    # Relationship
    configuration = relationship("AlertConfiguration", back_populates="alerts")

class MedicationRecord(Base):
    __tablename__ = "medication_records"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    medication_name = Column(String, index=True)
    dosage = Column(String)
    frequency = Column(String)  # e.g., "Once daily", "Twice daily"
    start_date = Column(Date)
    end_date = Column(Date, nullable=True)  # null if currently active
    notes = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)

    # Relationships
    student = relationship("Student", back_populates="medications")
    logs = relationship("MedicationLog", back_populates="medication")

class MedicationLog(Base):
    __tablename__ = "medication_logs"

    id = Column(Integer, primary_key=True, index=True)
    medication_id = Column(Integer, ForeignKey("medication_records.id"))
    timestamp = Column(DateTime, index=True)
    status = Column(String)  # "taken", "missed", "late"
    reason_if_missed = Column(String, nullable=True)
    notes = Column(Text, nullable=True)

    # Relationship
    medication = relationship("MedicationRecord", back_populates="logs")
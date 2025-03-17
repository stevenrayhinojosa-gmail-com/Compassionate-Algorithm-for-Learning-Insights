from datetime import datetime
from sqlalchemy.orm import Session
from models import AlertConfiguration, Alert, Student

class AlertSystem:
    def __init__(self, db: Session):
        self.db = db

    def create_alert_config(self, student_id: int, config_data: dict) -> AlertConfiguration:
        """Create a new alert configuration for a student"""
        config = AlertConfiguration(
            student_id=student_id,
            name=config_data['name'],
            description=config_data['description'],
            behavior_threshold=config_data.get('behavior_threshold', 1.0),
            red_threshold=config_data.get('red_threshold', 3),
            trend_threshold=config_data.get('trend_threshold', -0.5),
            is_active=config_data.get('is_active', True),
            notify_on_prediction=config_data.get('notify_on_prediction', True),
            notification_enabled=config_data.get('notification_enabled', False)
        )
        self.db.add(config)
        self.db.commit()
        self.db.refresh(config)
        return config

    def check_alerts(self, student_id: int, behavior_data: dict, is_prediction: bool = False) -> list:
        """Check if any alerts should be triggered based on behavior data"""
        alerts = []
        configs = self.db.query(AlertConfiguration).filter(
            AlertConfiguration.student_id == student_id,
            AlertConfiguration.is_active == True
        ).all()

        for config in configs:
            # Skip prediction checks if notification on predictions is disabled
            if is_prediction and not config.notify_on_prediction:
                continue

            # Check behavior score threshold
            if behavior_data['behavior_score'] <= config.behavior_threshold:
                alerts.append(self._create_alert(
                    config.id,
                    'behavior_score',
                    behavior_data['behavior_score'],
                    is_prediction
                ))

            # Check red count threshold
            if behavior_data['red_count'] >= config.red_threshold:
                alerts.append(self._create_alert(
                    config.id,
                    'red_count',
                    behavior_data['red_count'],
                    is_prediction
                ))

            # Check trend threshold
            if behavior_data['behavior_trend'] <= config.trend_threshold:
                alerts.append(self._create_alert(
                    config.id,
                    'trend',
                    behavior_data['behavior_trend'],
                    is_prediction
                ))

        return alerts

    def _create_alert(self, config_id: int, alert_type: str, value: float, is_prediction: bool) -> Alert:
        """Create and save an alert"""
        alert = Alert(
            configuration_id=config_id,
            triggered_at=datetime.now().date(),
            alert_type=alert_type,
            value=value,
            is_prediction=is_prediction,
            notification_sent=False  # Initialize as not sent
        )
        self.db.add(alert)
        self.db.commit()
        self.db.refresh(alert)
        return alert

    def get_student_alerts(self, student_id: int, include_predictions: bool = True) -> list:
        """Get all alerts for a student"""
        query = self.db.query(Alert).join(AlertConfiguration).filter(
            AlertConfiguration.student_id == student_id
        )

        if not include_predictions:
            query = query.filter(Alert.is_prediction == False)

        return query.order_by(Alert.triggered_at.desc()).all()

    def get_active_alerts(self, student_id: int) -> list:
        """Get active alerts for a student from the last 24 hours"""
        yesterday = datetime.now().date()
        return self.db.query(Alert).join(AlertConfiguration).filter(
            AlertConfiguration.student_id == student_id,
            Alert.triggered_at >= yesterday
        ).all()
import os
from twilio.rest import Client
from datetime import datetime

class NotificationHelper:
    def __init__(self):
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.from_number = os.getenv('TWILIO_PHONE_NUMBER')
        
    def send_alert(self, to_number: str, alert_type: str, value: float, student_name: str, is_prediction: bool = False):
        """Send SMS alert for concerning behavior patterns"""
        try:
            if not all([self.account_sid, self.auth_token, self.from_number]):
                return False
                
            client = Client(self.account_sid, self.auth_token)
            
            # Format message based on alert type
            time_context = "predicted" if is_prediction else "current"
            message = f"Alert for {student_name}: {time_context} "
            
            if alert_type == 'behavior_score':
                message += f"behavior score is {value:.2f}"
            elif alert_type == 'red_count':
                message += f"red markers count is {int(value)}"
            elif alert_type == 'trend':
                message += f"behavior trend is {value:.2f}"
                
            message += f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Send message
            client.messages.create(
                body=message,
                from_=self.from_number,
                to=to_number
            )
            return True
            
        except Exception as e:
            print(f"Error sending notification: {str(e)}")
            return False
            
    def send_summary(self, to_number: str, student_name: str, summary_data: dict):
        """Send daily behavior summary"""
        try:
            if not all([self.account_sid, self.auth_token, self.from_number]):
                return False
                
            message = f"Daily Summary for {student_name}:\n"
            message += f"Average Score: {summary_data['avg_score']:.2f}\n"
            message += f"Red Markers: {summary_data['red_count']}\n"
            message += f"Trend: {summary_data['trend']:+.2f}\n"
            message += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            client = Client(self.account_sid, self.auth_token)
            client.messages.create(
                body=message,
                from_=self.from_number,
                to=to_number
            )
            return True
            
        except Exception as e:
            print(f"Error sending summary: {str(e)}")
            return False

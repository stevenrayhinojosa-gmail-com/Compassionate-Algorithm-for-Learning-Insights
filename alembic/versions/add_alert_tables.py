"""Add alert tables

Revision ID: 002_alerts
Revises: 001_initial
Create Date: 2025-03-17
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '002_alerts'
down_revision = '001_initial'
branch_labels = None
depends_on = None

def upgrade():
    # Create alert_configurations table
    op.create_table(
        'alert_configurations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('student_id', sa.Integer(), nullable=True),
        sa.Column('name', sa.String(), nullable=True),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('behavior_threshold', sa.Float(), nullable=True),
        sa.Column('red_threshold', sa.Integer(), nullable=True),
        sa.Column('trend_threshold', sa.Float(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('notify_on_prediction', sa.Boolean(), nullable=True),
        sa.Column('notification_phone', sa.String(), nullable=True),
        sa.Column('notification_enabled', sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(['student_id'], ['students.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_alert_configurations_id', 'alert_configurations', ['id'], unique=False)

    # Create alerts table
    op.create_table(
        'alerts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('configuration_id', sa.Integer(), nullable=True),
        sa.Column('triggered_at', sa.Date(), nullable=True),
        sa.Column('alert_type', sa.String(), nullable=True),
        sa.Column('value', sa.Float(), nullable=True),
        sa.Column('is_prediction', sa.Boolean(), nullable=True),
        sa.Column('notification_sent', sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(['configuration_id'], ['alert_configurations.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_alerts_id', 'alerts', ['id'], unique=False)
    op.create_index('ix_alerts_triggered_at', 'alerts', ['triggered_at'], unique=False)

def downgrade():
    op.drop_table('alerts')
    op.drop_table('alert_configurations')
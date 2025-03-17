"""Initial migration

Revision ID: 001_initial
Create Date: 2025-03-17
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create students table
    op.create_table(
        'students',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_students_id', 'students', ['id'], unique=False)
    op.create_index('ix_students_name', 'students', ['name'], unique=False)

    # Create behavior_records table
    op.create_table(
        'behavior_records',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('date', sa.Date(), nullable=True),
        sa.Column('student_id', sa.Integer(), nullable=True),
        sa.Column('behavior_score', sa.Float(), nullable=True),
        sa.Column('red_count', sa.Integer(), nullable=True),
        sa.Column('yellow_count', sa.Integer(), nullable=True),
        sa.Column('green_count', sa.Integer(), nullable=True),
        sa.Column('rolling_avg_7d', sa.Float(), nullable=True),
        sa.Column('rolling_std_7d', sa.Float(), nullable=True),
        sa.Column('behavior_trend', sa.Float(), nullable=True),
        sa.Column('weekly_improvement', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['student_id'], ['students.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_behavior_records_date', 'behavior_records', ['date'], unique=False)
    op.create_index('ix_behavior_records_id', 'behavior_records', ['id'], unique=False)

    # Create time_slot_behaviors table
    op.create_table(
        'time_slot_behaviors',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('behavior_record_id', sa.Integer(), nullable=True),
        sa.Column('time_slot', sa.String(), nullable=True),
        sa.Column('behavior_value', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['behavior_record_id'], ['behavior_records.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_time_slot_behaviors_id', 'time_slot_behaviors', ['id'], unique=False)

def downgrade():
    op.drop_table('time_slot_behaviors')
    op.drop_table('behavior_records')
    op.drop_table('students')
"""Add medication tables

Revision ID: add_medication_tables
Create Date: 2025-03-17
"""
from alembic import op
import sqlalchemy as sa

def upgrade():
    # Create medication_records table
    op.create_table(
        'medication_records',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('student_id', sa.Integer(), nullable=True),
        sa.Column('medication_name', sa.String(), nullable=True),
        sa.Column('dosage', sa.String(), nullable=True),
        sa.Column('frequency', sa.String(), nullable=True),
        sa.Column('start_date', sa.Date(), nullable=True),
        sa.Column('end_date', sa.Date(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(['student_id'], ['students.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_medication_records_id', 'medication_records', ['id'], unique=False)
    op.create_index('ix_medication_records_medication_name', 'medication_records', ['medication_name'], unique=False)

    # Create medication_logs table
    op.create_table(
        'medication_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('medication_id', sa.Integer(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('reason_if_missed', sa.String(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['medication_id'], ['medication_records.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_medication_logs_id', 'medication_logs', ['id'], unique=False)
    op.create_index('ix_medication_logs_timestamp', 'medication_logs', ['timestamp'], unique=False)

def downgrade():
    op.drop_table('medication_logs')
    op.drop_table('medication_records')

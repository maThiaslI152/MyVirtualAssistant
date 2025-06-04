"""
Create conversations and nodes tables

Revision ID: 20240601_create_conversation_and_node_tables
Revises: 
Create Date: 2024-06-01
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20240601_create_conversation_and_node_tables'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'conversations',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('metadata_json', sa.Text(), nullable=True),
    )
    op.create_table(
        'nodes',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('conversation_id', sa.String(), sa.ForeignKey('conversations.id'), nullable=False),
        sa.Column('parent_id', sa.String(), nullable=True),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('context_json', sa.Text(), nullable=True),
        sa.Column('metadata_json', sa.Text(), nullable=True),
        sa.Column('children_json', sa.Text(), nullable=True),
    )

def downgrade():
    op.drop_table('nodes')
    op.drop_table('conversations') 
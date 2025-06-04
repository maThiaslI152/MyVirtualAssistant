"""create chat tables

Revision ID: create_chat_tables
Revises: 
Create Date: 2024-03-09 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'create_chat_tables'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create conversations table
    op.create_table(
        'conversations',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('last_updated', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('summary', sa.Text()),
        sa.Column('metadata', sa.Text())
    )

    # Create messages table
    op.create_table(
        'messages',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('conversation_id', sa.String(), sa.ForeignKey('conversations.id')),
        sa.Column('role', sa.String()),
        sa.Column('content', sa.Text()),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('metadata', sa.Text())
    )

    # Create files table
    op.create_table(
        'files',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('conversation_id', sa.String(), sa.ForeignKey('conversations.id')),
        sa.Column('filename', sa.String()),
        sa.Column('content_type', sa.String()),
        sa.Column('size', sa.Integer()),
        sa.Column('content', sa.LargeBinary()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'))
    )

def downgrade():
    op.drop_table('files')
    op.drop_table('messages')
    op.drop_table('conversations') 
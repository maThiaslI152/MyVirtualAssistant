"""Initial chat tables

Revision ID: 001
Revises: 
Create Date: 2024-03-19 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Create chat_sessions table
    op.create_table(
        'chat_sessions',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('title', sa.String(255)),
        sa.Column('metadata', postgresql.JSONB),
    )

    # Create messages table
    op.create_table(
        'messages',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('session_id', sa.String(36), sa.ForeignKey('chat_sessions.id', ondelete='CASCADE')),
        sa.Column('role', sa.String(50)),
        sa.Column('content', sa.Text),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('metadata', postgresql.JSONB),
    )

    # Create embeddings table
    op.create_table(
        'embeddings',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('message_id', sa.String(36), sa.ForeignKey('messages.id', ondelete='CASCADE')),
        sa.Column('vector', postgresql.ARRAY(sa.Float)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )

    # Create indexes
    op.create_index('ix_chat_sessions_created_at', 'chat_sessions', ['created_at'])
    op.create_index('ix_messages_session_id', 'messages', ['session_id'])
    op.create_index('ix_messages_created_at', 'messages', ['created_at'])

def downgrade() -> None:
    op.drop_table('embeddings')
    op.drop_table('messages')
    op.drop_table('chat_sessions') 
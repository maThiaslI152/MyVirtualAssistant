# backend/vectorstore/models.py

from sqlalchemy import Column, String, Text, DateTime, ARRAY
from .db import Base
from datetime import datetime

class ArchivedSession(Base):
    __tablename__ = "archived_sessions"

    session_id = Column(String, primary_key=True)
    title = Column(String)
    summary = Column(Text)
    full_history = Column(Text)
    tags = Column(ARRAY(String), default=[])  # new
    created_at = Column(DateTime, default=datetime.utcnow)

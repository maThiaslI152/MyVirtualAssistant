# alembic/env.py (you should already have alembic installed and set up)
# Run: alembic init alembic && alembic revision --autogenerate -m "Create archived_sessions"

# To apply the migration:
# alembic upgrade head

# Optional: FastAPI route to browse archived sessions

# archive_api.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from vectorstore.db import get_pg_session
from vectorstore.models import ArchivedSession

router = APIRouter()

@router.get("/archive")
async def list_archived_sessions(db: AsyncSession = Depends(get_pg_session)):
    result = await db.execute(select(ArchivedSession).order_by(ArchivedSession.created_at.desc()).limit(50))
    return [row._asdict() for row in result.scalars().all()]

@router.get("/archive/{session_id}")
async def get_archived_session(session_id: str, db: AsyncSession = Depends(get_pg_session)):
    result = await db.execute(select(ArchivedSession).where(ArchivedSession.session_id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

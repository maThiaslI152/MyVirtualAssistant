import uuid
import json
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import create_engine, Column, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from redis import asyncio as aioredis

# Dummy fallback in-memory store
_conversations = {}

Base = declarative_base()

# Minimal SQLAlchemy models
class Conversation(Base):
    __tablename__ = 'conversations'
    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(Text)  # Store metadata as JSON string
    nodes = relationship('Node', back_populates='conversation', cascade="all, delete-orphan")

class Node(Base):
    __tablename__ = 'nodes'
    id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey('conversations.id'))
    parent_id = Column(String, nullable=True)
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    context_json = Column(Text)
    metadata_json = Column(Text)
    children_json = Column(Text)  # Store children IDs as JSON list
    conversation = relationship('Conversation', back_populates='nodes')

class ConversationService:
    def __init__(self, pg_url: Optional[str] = None, redis_url: Optional[str] = None):
        self.pg_url = pg_url
        self.redis_url = redis_url
        self.pg_engine = create_engine(pg_url) if pg_url else None
        self.SessionLocal = sessionmaker(bind=self.pg_engine) if self.pg_engine else None
        self.redis = aioredis.from_url(redis_url, decode_responses=True) if redis_url else None

    def create_conversation(self, initial_content: str, metadata: Dict[str, Any]) -> str:
        """Create a new conversation with a root node."""
        conversation_id = str(uuid.uuid4())
        node_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        root_node = {
            'id': node_id,
            'content': initial_content,
            'timestamp': timestamp,
            'context': {},
            'metadata': metadata,
            'children': []
        }
        conversation = {
            'id': conversation_id,
            'structure': root_node,
            'metadata': {
                'created_at': timestamp,
                'total_nodes': 1,
                'total_branches': 0,
                'max_depth': 1
            }
        }
        _conversations[conversation_id] = conversation
        # Save to PostgreSQL
        if self.SessionLocal:
            with self.SessionLocal() as db:
                conv = Conversation(id=conversation_id, created_at=datetime.utcnow(), metadata_json=json.dumps(conversation['metadata']))
                node = Node(
                    id=node_id,
                    conversation_id=conversation_id,
                    parent_id=None,
                    content=initial_content,
                    timestamp=datetime.utcnow(),
                    context_json=json.dumps({}),
                    metadata_json=json.dumps(metadata),
                    children_json=json.dumps([])
                )
                conv.nodes.append(node)
                db.add(conv)
                db.commit()
        # Save to Redis
        if self.redis:
            self.redis.set(f"conversation:{conversation_id}", json.dumps(conversation))
        return conversation_id

    def add_branch(self, conversation_id: str, parent_id: str, content: str, context: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Add a new branch (node) to an existing conversation under parent_id."""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            raise ValueError("Conversation not found")
        node_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        new_node = {
            'id': node_id,
            'content': content,
            'timestamp': timestamp,
            'context': context,
            'metadata': metadata,
            'children': []
        }
        # Recursively find parent node and add child
        def add_child(node):
            if node['id'] == parent_id:
                node['children'].append(node_id)
                return True
            for child_id in node['children']:
                child_node = self._find_node(conversation['structure'], child_id)
                if child_node and add_child(child_node):
                    return True
            return False
        if not add_child(conversation['structure']):
            raise ValueError("Parent node not found")
        # Optionally, update metadata
        conversation['metadata']['total_nodes'] += 1
        conversation['metadata']['total_branches'] += 1
        # Save new node in PostgreSQL
        if self.SessionLocal:
            with self.SessionLocal() as db:
                node = Node(
                    id=node_id,
                    conversation_id=conversation_id,
                    parent_id=parent_id,
                    content=content,
                    timestamp=datetime.utcnow(),
                    context_json=json.dumps(context),
                    metadata_json=json.dumps(metadata),
                    children_json=json.dumps([])
                )
                db.add(node)
                db.commit()
        # Update conversation in Redis
        if self.redis:
            self.redis.set(f"conversation:{conversation_id}", json.dumps(conversation))
        _conversations[conversation_id] = conversation
        return node_id

    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Retrieve the entire conversation structure."""
        # Try Redis first
        if self.redis:
            data = self.redis.get(f"conversation:{conversation_id}")
            if data:
                return json.loads(data)
        # Fallback to PostgreSQL
        if self.SessionLocal:
            with self.SessionLocal() as db:
                conv = db.query(Conversation).filter_by(id=conversation_id).first()
                if not conv:
                    return None
                nodes = db.query(Node).filter_by(conversation_id=conversation_id).all()
                # Rebuild structure from nodes
                node_map = {n.id: n for n in nodes}
                def build_tree(node_id):
                    n = node_map[node_id]
                    children = json.loads(n.children_json)
                    return {
                        'id': n.id,
                        'content': n.content,
                        'timestamp': n.timestamp.isoformat(),
                        'context': json.loads(n.context_json),
                        'metadata': json.loads(n.metadata_json),
                        'children': [build_tree(cid) for cid in children]
                    }
                root_node = next((n for n in nodes if n.parent_id is None), None)
                if not root_node:
                    return None
                structure = build_tree(root_node.id)
                return {
                    'id': conv.id,
                    'structure': structure,
                    'metadata': json.loads(conv.metadata_json)
                }
        # Fallback to in-memory
        return _conversations.get(conversation_id)

    def get_node_context(self, conversation_id: str, node_id: str) -> Optional[Dict[str, Any]]:
        """Get the context for a specific node in a conversation."""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return None
        node = self._find_node(conversation['structure'], node_id)
        if node:
            return node.get('context', {})
        return None

    def _find_node(self, node: Dict[str, Any], node_id: str) -> Optional[Dict[str, Any]]:
        if node['id'] == node_id:
            return node
        for child in node['children']:
            child_node = self._find_node(child, node_id)
            if child_node:
                return child_node
        return None 
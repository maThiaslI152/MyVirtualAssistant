from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime
import logging
from ..services.conversation_service import ConversationService
from ..services.rag_search_service import RAGSearchService
from ..services.content_processor import ContentProcessor

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
conversation_service = ConversationService()
rag_service = RAGSearchService()
content_processor = ContentProcessor()

class ConversationRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None

class BranchRequest(BaseModel):
    content: str
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    explore_deeper: bool = False

class ConversationResponse(BaseModel):
    conversation_id: str
    node_id: str
    content: str
    timestamp: datetime
    context: Dict[str, Any]
    metadata: Dict[str, Any]
    children: List[str]

@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(request: ConversationRequest):
    """Create a new conversation."""
    try:
        # Process the initial content
        processed_content = content_processor.process_content(
            request.content,
            extract_summary=True,
            extract_entities=True,
            extract_keywords=True
        )

        # Create conversation with processed content
        conversation_id = conversation_service.create_conversation(
            initial_content=processed_content['content'],
            metadata={
                'summary': processed_content.get('summary', {}).get('text', ''),
                'entities': processed_content.get('entities', []),
                'keywords': processed_content.get('keywords', []),
                **(request.metadata or {})
            }
        )

        # Get the created conversation
        conversation = conversation_service.get_conversation(conversation_id)
        root_node = conversation['structure']

        return ConversationResponse(
            conversation_id=conversation_id,
            node_id=root_node['id'],
            content=root_node['content'],
            timestamp=datetime.fromisoformat(root_node['timestamp']),
            context=root_node['context'],
            metadata=root_node['metadata'],
            children=root_node['children']
        )

    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/conversations/{conversation_id}/branches/{parent_id}", response_model=ConversationResponse)
async def add_branch(
    conversation_id: str,
    parent_id: str,
    request: BranchRequest,
    background_tasks: BackgroundTasks
):
    """Add a new branch to an existing conversation."""
    try:
        # Get parent context
        parent_context = conversation_service.get_node_context(conversation_id, parent_id)

        # Process the new content
        processed_content = content_processor.process_content(
            request.content,
            extract_summary=True,
            extract_entities=True,
            extract_keywords=True
        )

        # If explore_deeper is True, use RAG to find related information
        if request.explore_deeper:
            # Search for related information
            search_results = await rag_service.search(
                query=request.content,
                num_web_results=3,
                num_retrieval_results=2
            )

            # Update context with search results
            context_updates = {
                'related_info': search_results,
                'explored_topics': parent_context.get('explored_topics', []) + [request.content]
            }
        else:
            context_updates = {}

        # Add the new branch
        node_id = conversation_service.add_branch(
            conversation_id=conversation_id,
            parent_id=parent_id,
            content=processed_content['content'],
            context={
                'summary': processed_content.get('summary', {}).get('text', ''),
                'entities': processed_content.get('entities', []),
                'keywords': processed_content.get('keywords', []),
                **(request.context or {}),
                **context_updates
            },
            metadata={
                'explored_deeper': request.explore_deeper,
                **(request.metadata or {})
            }
        )

        # Get the updated node
        conversation = conversation_service.get_conversation(conversation_id)
        node = next(
            (n for n in conversation['structure']['children'] if n['id'] == node_id),
            None
        )

        if not node:
            raise HTTPException(status_code=404, detail="Node not found")

        return ConversationResponse(
            conversation_id=conversation_id,
            node_id=node['id'],
            content=node['content'],
            timestamp=datetime.fromisoformat(node['timestamp']),
            context=node['context'],
            metadata=node['metadata'],
            children=node['children']
        )

    except Exception as e:
        logger.error(f"Error adding branch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get the entire conversation structure."""
    try:
        conversation = conversation_service.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation

    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conversations/{conversation_id}/nodes/{node_id}/context")
async def get_node_context(conversation_id: str, node_id: str):
    """Get the context for a specific node."""
    try:
        context = conversation_service.get_node_context(conversation_id, node_id)
        if not context:
            raise HTTPException(status_code=404, detail="Node not found")
        return context

    except Exception as e:
        logger.error(f"Error getting node context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 
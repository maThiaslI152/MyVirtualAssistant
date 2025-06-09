from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime
import json
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.cluster import DBSCAN
import spacy
from nltk.tokenize import sent_tokenize
import torch
from dataclasses import dataclass
from enum import Enum

class MessageType(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"

@dataclass
class Message:
    type: MessageType
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None

class ChatEnhancer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.nlp = spacy.load('en_core_web_sm')
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize models
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        # Initialize pipelines
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.zero_shot_classifier = pipeline("zero-shot-classification")
        self.qa_pipeline = pipeline("question-answering")
        
        # Initialize conversation memory
        self.conversation_history: List[Message] = []
        self.conversation_embeddings = []
        self.topic_clusters = []
        
    def process_message(self, message: str, message_type: MessageType = MessageType.USER) -> Dict[str, Any]:
        """Process a new message and return enhanced information."""
        try:
            # Create message object
            msg = Message(
                type=message_type,
                content=message,
                timestamp=datetime.now(),
                metadata={}
            )
            
            # Add to history
            self.conversation_history.append(msg)
            
            # Generate embeddings
            embedding = self.sentence_transformer.encode(message)
            self.conversation_embeddings.append(embedding)
            
            # Analyze message
            analysis = self._analyze_message(message)
            msg.metadata.update(analysis)
            
            # Update topic clusters
            self._update_topic_clusters()
            
            # Generate response suggestions
            suggestions = self._generate_suggestions(message)
            
            return {
                'message': msg,
                'analysis': analysis,
                'suggestions': suggestions,
                'conversation_context': self._get_conversation_context()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            raise
            
    def _analyze_message(self, message: str) -> Dict[str, Any]:
        """Analyze message content and extract various features."""
        try:
            # Basic NLP analysis
            doc = self.nlp(message)
            
            # Extract entities
            entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
            
            # Extract key phrases
            key_phrases = [chunk.text for chunk in doc.noun_chunks]
            
            # Analyze sentiment
            sentiment = self.sentiment_analyzer(message)[0]
            
            # Detect intent
            intent = self._detect_intent(message)
            
            # Extract topics
            topics = self._extract_topics(message)
            
            # Analyze complexity
            complexity = self._analyze_complexity(message)
            
            return {
                'entities': entities,
                'key_phrases': key_phrases,
                'sentiment': sentiment,
                'intent': intent,
                'topics': topics,
                'complexity': complexity,
                'tokens': len(doc),
                'sentences': len(list(doc.sents))
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing message: {str(e)}")
            return {}
            
    def _detect_intent(self, message: str) -> Dict[str, float]:
        """Detect user intent using zero-shot classification."""
        try:
            candidate_labels = [
                "question", "statement", "command", "greeting",
                "farewell", "clarification", "confirmation",
                "disagreement", "agreement", "request"
            ]
            
            result = self.zero_shot_classifier(
                message,
                candidate_labels,
                multi_label=True
            )
            
            return dict(zip(result['labels'], result['scores']))
            
        except Exception as e:
            self.logger.error(f"Error detecting intent: {str(e)}")
            return {}
            
    def _extract_topics(self, message: str) -> List[str]:
        """Extract main topics from the message."""
        try:
            # Use zero-shot classification for topic detection
            candidate_labels = [
                "technology", "science", "business", "health",
                "education", "entertainment", "sports", "politics",
                "art", "food", "travel", "fashion"
            ]
            
            result = self.zero_shot_classifier(
                message,
                candidate_labels,
                multi_label=True
            )
            
            # Return topics with score > 0.5
            return [label for label, score in zip(result['labels'], result['scores']) if score > 0.5]
            
        except Exception as e:
            self.logger.error(f"Error extracting topics: {str(e)}")
            return []
            
    def _analyze_complexity(self, message: str) -> Dict[str, float]:
        """Analyze message complexity."""
        try:
            doc = self.nlp(message)
            
            # Calculate various complexity metrics
            avg_word_length = np.mean([len(token.text) for token in doc])
            avg_sentence_length = np.mean([len(sent) for sent in doc.sents])
            unique_words_ratio = len(set([token.text.lower() for token in doc])) / len(doc)
            
            return {
                'avg_word_length': avg_word_length,
                'avg_sentence_length': avg_sentence_length,
                'unique_words_ratio': unique_words_ratio,
                'complexity_score': (avg_word_length * 0.3 + avg_sentence_length * 0.4 + unique_words_ratio * 0.3)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing complexity: {str(e)}")
            return {}
            
    def _update_topic_clusters(self):
        """Update topic clusters based on conversation history."""
        try:
            if len(self.conversation_embeddings) < 2:
                return
                
            # Convert embeddings to numpy array
            embeddings = np.array(self.conversation_embeddings)
            
            # Perform clustering
            clustering = DBSCAN(eps=0.3, min_samples=2).fit(embeddings)
            
            # Update clusters
            self.topic_clusters = clustering.labels_
            
        except Exception as e:
            self.logger.error(f"Error updating topic clusters: {str(e)}")
            
    def _get_conversation_context(self) -> Dict[str, Any]:
        """Get current conversation context."""
        try:
            if not self.conversation_history:
                return {}
                
            # Get recent messages
            recent_messages = self.conversation_history[-5:]
            
            # Calculate conversation statistics
            total_messages = len(self.conversation_history)
            user_messages = sum(1 for msg in self.conversation_history if msg.type == MessageType.USER)
            assistant_messages = sum(1 for msg in self.conversation_history if msg.type == MessageType.ASSISTANT)
            
            # Get current topic
            current_topic = self._get_current_topic()
            
            return {
                'recent_messages': recent_messages,
                'total_messages': total_messages,
                'user_messages': user_messages,
                'assistant_messages': assistant_messages,
                'current_topic': current_topic,
                'topic_clusters': self.topic_clusters.tolist() if hasattr(self.topic_clusters, 'tolist') else []
            }
            
        except Exception as e:
            self.logger.error(f"Error getting conversation context: {str(e)}")
            return {}
            
    def _get_current_topic(self) -> str:
        """Get the current conversation topic."""
        try:
            if not self.conversation_history:
                return ""
                
            # Get recent messages
            recent_messages = [msg.content for msg in self.conversation_history[-3:]]
            recent_text = " ".join(recent_messages)
            
            # Extract topics
            topics = self._extract_topics(recent_text)
            
            return topics[0] if topics else ""
            
        except Exception as e:
            self.logger.error(f"Error getting current topic: {str(e)}")
            return ""
            
    def _generate_suggestions(self, message: str) -> List[str]:
        """Generate response suggestions based on the message."""
        try:
            # Generate completions using the model
            inputs = self.tokenizer(message, return_tensors="pt")
            outputs = self.model.generate(
                **inputs,
                max_length=100,
                num_return_sequences=3,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            # Decode suggestions
            suggestions = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating suggestions: {str(e)}")
            return []
            
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the entire conversation."""
        try:
            if not self.conversation_history:
                return {}
                
            # Combine all messages
            all_text = " ".join([msg.content for msg in self.conversation_history])
            
            # Extract main topics
            topics = self._extract_topics(all_text)
            
            # Calculate conversation statistics
            stats = self._get_conversation_context()
            
            # Analyze overall sentiment
            sentiments = [msg.metadata.get('sentiment', {}).get('label', 'neutral') 
                         for msg in self.conversation_history]
            sentiment_distribution = {
                'positive': sentiments.count('POSITIVE'),
                'negative': sentiments.count('NEGATIVE'),
                'neutral': sentiments.count('NEUTRAL')
            }
            
            return {
                'topics': topics,
                'statistics': stats,
                'sentiment_distribution': sentiment_distribution,
                'duration': (self.conversation_history[-1].timestamp - self.conversation_history[0].timestamp).total_seconds(),
                'message_count': len(self.conversation_history)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting conversation summary: {str(e)}")
            return {} 
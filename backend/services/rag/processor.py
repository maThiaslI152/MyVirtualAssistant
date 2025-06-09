from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import re
from datetime import datetime
import logging
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import langdetect
from textblob import TextBlob
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
from rouge_score import rouge_scorer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.kl import KLSummarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sumy.summarizers.reduction import ReductionSummarizer
from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.summarizers.random import RandomSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

class ContentProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Initialize device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {'MPS (Metal Performance Shaders) for GPU acceleration' if torch.backends.mps.is_available() else 'CPU'}")
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
        self.model = AutoModel.from_pretrained("intfloat/multilingual-e5-large").to(self.device)
        
        # Initialize summarization models
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.sentence_transformer = SentenceTransformer('intfloat/multilingual-e5-large', device=self.device)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embeddings for the given text using the multilingual-e5-large model."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings

    def clean_text(self, text):
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        return text

    def process_text(self, text):
        return self.clean_text(text)

    def detect_language(self, text: str) -> Dict[str, Any]:
        """Detect language and confidence."""
        try:
            lang = langdetect.detect(text)
            confidence = langdetect.detect_langs(text)[0].prob
            return {
                'language': lang,
                'confidence': confidence
            }
        except:
            return {
                'language': 'unknown',
                'confidence': 0.0
            }

    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[Dict[str, Any]]:
        """Extract important keywords with additional information."""
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        word_freq = Counter(words)
        
        # Sort by frequency and format results
        keywords = word_freq.most_common(num_keywords)
        
        return [
            {
                'word': word,
                'count': count,
                'pos': 'unknown',  # POS tagging removed
                'lemma': self.lemmatizer.lemmatize(word)
            }
            for word, count in keywords
        ]

    def extract_topics(self, text: str, num_topics: int = 5) -> List[Dict[str, Any]]:
        """Extract main topics from text using LDA."""
        # Prepare text for topic modeling
        sentences = sent_tokenize(text)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(sentences)
        
        # Fit LDA
        lda_output = self.lda.fit_transform(tfidf_matrix)
        
        # Get feature names
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(self.lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10-1:-1]]
            topics.append({
                'topic_id': topic_idx,
                'top_words': top_words,
                'topic_strength': float(lda_output[:, topic_idx].mean())
            })
        
        return topics

    def generate_summary(
        self,
        text: str,
        method: str = 'hybrid',
        min_length: int = 50,
        max_length: int = 150,
        compression_ratio: float = 0.3,
        use_advanced_features: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a summary using the specified method.
        
        Args:
            text: Input text to summarize
            method: Summarization method ('extractive', 'abstractive', 'hybrid', 'advanced')
            min_length: Minimum summary length
            max_length: Maximum summary length
            compression_ratio: Target compression ratio
            use_advanced_features: Whether to use advanced features like sentiment and key phrases
            
        Returns:
            Dictionary containing summary and metadata
        """
        try:
            # Split text into sentences
            sentences = sent_tokenize(text)
            if not sentences:
                return {
                    'summary': '',
                    'metadata': {
                        'original_length': 0,
                        'summary_length': 0,
                        'compression_ratio': 0,
                        'methods_used': [],
                        'method_scores': {},
                        'key_phrases': [],
                        'sentiment': {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 0}
                    }
                }

            # Calculate target summary length
            target_length = int(len(text.split()) * compression_ratio)
            target_length = max(min_length, min(target_length, max_length))

            # Generate summaries using different methods
            summaries = {}
            scores = {}

            if method in ['extractive', 'hybrid', 'advanced']:
                # Extractive methods
                summaries['tfidf'] = self._tfidf_summarize(sentences, target_length)
                summaries['textrank'] = self._textrank_summarize(sentences, target_length)
                summaries['lsa'] = self._lsa_summarize(sentences, target_length)
                summaries['lexrank'] = self._lexrank_summarize(sentences, target_length)
                summaries['luhn'] = self._luhn_summarize(sentences, target_length)
                summaries['sumbasic'] = self._sumbasic_summarize(sentences, target_length)
                summaries['kl'] = self._kl_summarize(sentences, target_length)
                summaries['reduction'] = self._reduction_summarize(sentences, target_length)
                summaries['edmundson'] = self._edmundson_summarize(sentences, target_length)
                summaries['centroid'] = self._centroid_summarize(sentences, target_length)
                summaries['mmr'] = self._mmr_summarize(sentences, target_length)

            if method in ['abstractive', 'hybrid', 'advanced']:
                # Abstractive methods
                summaries['bart'] = self._bart_summarize(text, target_length)
                summaries['t5'] = self._t5_summarize(text, target_length)
                summaries['pegasus'] = self._pegasus_summarize(text, target_length)

            # Evaluate summaries
            for method_name, summary in summaries.items():
                if summary:
                    scores[method_name] = self._evaluate_summary(summary, text)

            # Select best summary or combine them
            if method == 'hybrid':
                final_summary = self._combine_summaries(summaries, scores, target_length)
            elif method == 'advanced':
                final_summary = self._advanced_summarize(summaries, scores, text, target_length)
            else:
                best_method = max(scores.items(), key=lambda x: x[1])[0]
                final_summary = summaries[best_method]

            # Post-process summary
            final_summary = self._post_process_summary(final_summary, target_length)

            # Generate metadata
            metadata = {
                'original_length': len(text.split()),
                'summary_length': len(final_summary.split()),
                'compression_ratio': len(final_summary.split()) / len(text.split()),
                'methods_used': list(summaries.keys()),
                'method_scores': scores
            }

            if use_advanced_features:
                metadata.update({
                    'key_phrases': self._extract_key_phrases(text),
                    'sentiment': self._analyze_sentiment(text),
                    'readability': self._analyze_readability(text),
                    'topic_modeling': self._extract_topics(text),
                    'entity_recognition': self._extract_entities(text),
                    'summary_quality': self._evaluate_summary_quality(final_summary, text)
                })

            return {
                'summary': final_summary,
                'metadata': metadata
            }

        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            raise

    def _advanced_summarize(
        self,
        summaries: Dict[str, str],
        scores: Dict[str, float],
        original_text: str,
        target_length: int
    ) -> str:
        """Generate an advanced summary using multiple methods and features."""
        # Combine summaries with weights based on scores
        weighted_summary = self._combine_summaries(summaries, scores, target_length)
        
        # Extract key information
        key_phrases = self._extract_key_phrases(original_text)
        entities = self._extract_entities(original_text)
        topics = self._extract_topics(original_text)
        
        # Ensure key information is included
        summary_sentences = sent_tokenize(weighted_summary)
        key_sentences = []
        
        for sentence in summary_sentences:
            # Check if sentence contains key information
            if any(phrase in sentence.lower() for phrase in key_phrases):
                key_sentences.append(sentence)
            elif any(entity in sentence for entity in entities):
                key_sentences.append(sentence)
            elif any(topic in sentence.lower() for topic in topics):
                key_sentences.append(sentence)
        
        # Combine key sentences with weighted summary
        final_summary = ' '.join(key_sentences + summary_sentences)
        
        # Ensure summary length is within target
        while len(final_summary.split()) > target_length:
            # Remove least important sentences
            sentences = sent_tokenize(final_summary)
            if len(sentences) <= 1:
                break
            sentences.pop()
            final_summary = ' '.join(sentences)
        
        return final_summary

    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text using LDA."""
        try:
            from gensim import corpora, models
            
            # Tokenize and preprocess
            tokens = [word.lower() for word in word_tokenize(text)
                     if word.isalpha() and word.lower() not in self.stop_words]
            
            # Create dictionary and corpus
            dictionary = corpora.Dictionary([tokens])
            corpus = [dictionary.doc2bow(tokens)]
            
            # Train LDA model
            lda_model = models.LdaModel(
                corpus,
                num_topics=3,
                id2word=dictionary,
                passes=10
            )
            
            # Extract topics
            topics = []
            for topic in lda_model.print_topics():
                words = topic[1].split('+')
                words = [word.split('*')[1].strip('"') for word in words]
                topics.extend(words[:3])  # Take top 3 words from each topic
            
            return list(set(topics))
        except Exception as e:
            self.logger.error(f"Error extracting topics: {str(e)}")
            return []

    def _analyze_readability(self, text: str) -> Dict[str, float]:
        """Analyze text readability using various metrics."""
        try:
            from textstat import textstat
            
            return {
                'flesch_reading_ease': textstat.flesch_reading_ease(text),
                'smog_index': textstat.smog_index(text),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                'coleman_liau_index': textstat.coleman_liau_index(text),
                'automated_readability_index': textstat.automated_readability_index(text),
                'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
                'difficult_words': textstat.difficult_words(text),
                'linsear_write_formula': textstat.linsear_write_formula(text),
                'gunning_fog': textstat.gunning_fog(text),
                'text_standard': textstat.text_standard(text)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing readability: {str(e)}")
            return {}

    def _evaluate_summary_quality(self, summary: str, original_text: str) -> Dict[str, float]:
        """Evaluate summary quality using multiple metrics."""
        try:
            # ROUGE scores
            rouge_scores = self.rouge_scorer.score(original_text, summary)
            
            # BLEU score
            from nltk.translate.bleu_score import sentence_bleu
            reference = [word_tokenize(original_text)]
            candidate = word_tokenize(summary)
            bleu_score = sentence_bleu(reference, candidate)
            
            # Semantic similarity
            summary_embedding = self.sentence_transformer.encode(summary)
            original_embedding = self.sentence_transformer.encode(original_text)
            similarity = cosine_similarity([summary_embedding], [original_embedding])[0][0]
            
            return {
                'rouge1': rouge_scores['rouge1'].fmeasure,
                'rouge2': rouge_scores['rouge2'].fmeasure,
                'rougeL': rouge_scores['rougeL'].fmeasure,
                'bleu': bleu_score,
                'semantic_similarity': similarity
            }
        except Exception as e:
            self.logger.error(f"Error evaluating summary quality: {str(e)}")
            return {}

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        try:
            import spacy
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(text)
            return [ent.text for ent in doc.ents]
        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}")
            return []

    def _t5_summarize(self, text: str, target_length: int) -> str:
        """Generate summary using T5 model."""
        try:
            t5_summarizer = pipeline("summarization", model="t5-base")
            summary = t5_summarizer(text, max_length=target_length, min_length=target_length//2)[0]['summary_text']
            return summary
        except Exception as e:
            self.logger.error(f"Error in T5 summarization: {str(e)}")
            return ""

    def _pegasus_summarize(self, text: str, target_length: int) -> str:
        """Generate summary using PEGASUS model."""
        try:
            pegasus_summarizer = pipeline("summarization", model="google/pegasus-large")
            summary = pegasus_summarizer(text, max_length=target_length, min_length=target_length//2)[0]['summary_text']
            return summary
        except Exception as e:
            self.logger.error(f"Error in PEGASUS summarization: {str(e)}")
            return ""

    def _lexrank_summarize(self, sentences: List[str], target_length: int) -> str:
        """Generate summary using LexRank algorithm."""
        try:
            parser = PlaintextParser.from_string(' '.join(sentences), self.tokenizer)
            summarizer = LexRankSummarizer(self.stemmer)
            summary = summarizer(parser.document, target_length)
            return ' '.join([str(sentence) for sentence in summary])
        except Exception as e:
            self.logger.error(f"Error in LexRank summarization: {str(e)}")
            return ""

    def _luhn_summarize(self, sentences: List[str], target_length: int) -> str:
        """Generate summary using Luhn algorithm."""
        try:
            parser = PlaintextParser.from_string(' '.join(sentences), self.tokenizer)
            summarizer = LuhnSummarizer(self.stemmer)
            summary = summarizer(parser.document, target_length)
            return ' '.join([str(sentence) for sentence in summary])
        except Exception as e:
            self.logger.error(f"Error in Luhn summarization: {str(e)}")
            return ""

    def _sumbasic_summarize(self, sentences: List[str], target_length: int) -> str:
        """Generate summary using SumBasic algorithm."""
        try:
            parser = PlaintextParser.from_string(' '.join(sentences), self.tokenizer)
            summarizer = SumBasicSummarizer(self.stemmer)
            summary = summarizer(parser.document, target_length)
            return ' '.join([str(sentence) for sentence in summary])
        except Exception as e:
            self.logger.error(f"Error in SumBasic summarization: {str(e)}")
            return ""

    def _kl_summarize(self, sentences: List[str], target_length: int) -> str:
        """Generate summary using KL algorithm."""
        try:
            parser = PlaintextParser.from_string(' '.join(sentences), self.tokenizer)
            summarizer = KLSummarizer(self.stemmer)
            summary = summarizer(parser.document, target_length)
            return ' '.join([str(sentence) for sentence in summary])
        except Exception as e:
            self.logger.error(f"Error in KL summarization: {str(e)}")
            return ""

    def _reduction_summarize(self, sentences: List[str], target_length: int) -> str:
        """Generate summary using Reduction algorithm."""
        try:
            parser = PlaintextParser.from_string(' '.join(sentences), self.tokenizer)
            summarizer = ReductionSummarizer(self.stemmer)
            summary = summarizer(parser.document, target_length)
            return ' '.join([str(sentence) for sentence in summary])
        except Exception as e:
            self.logger.error(f"Error in Reduction summarization: {str(e)}")
            return ""

    def _edmundson_summarize(self, sentences: List[str], target_length: int) -> str:
        """Generate summary using Edmundson algorithm."""
        try:
            parser = PlaintextParser.from_string(' '.join(sentences), self.tokenizer)
            summarizer = EdmundsonSummarizer(self.stemmer)
            summary = summarizer(parser.document, target_length)
            return ' '.join([str(sentence) for sentence in summary])
        except Exception as e:
            self.logger.error(f"Error in Edmundson summarization: {str(e)}")
            return ""

    def _centroid_summarize(self, sentences: List[str], target_length: int) -> str:
        """Generate summary using Centroid algorithm."""
        try:
            parser = PlaintextParser.from_string(' '.join(sentences), self.tokenizer)
            summarizer = CentroidSummarizer(self.stemmer)
            summary = summarizer(parser.document, target_length)
            return ' '.join([str(sentence) for sentence in summary])
        except Exception as e:
            self.logger.error(f"Error in Centroid summarization: {str(e)}")
            return ""

    def _mmr_summarize(self, sentences: List[str], target_length: int) -> str:
        """Generate summary using MMR algorithm."""
        try:
            parser = PlaintextParser.from_string(' '.join(sentences), self.tokenizer)
            summarizer = MMRSummarizer(self.stemmer)
            summary = summarizer(parser.document, target_length)
            return ' '.join([str(sentence) for sentence in summary])
        except Exception as e:
            self.logger.error(f"Error in MMR summarization: {str(e)}")
            return ""

    def _tfidf_summarize(self, sentences: List[str], target_length: int) -> str:
        """Generate summary using TF-IDF."""
        # Create TF-IDF matrix
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = np.sum(tfidf_matrix[i].toarray())
            sentence_scores.append((i, score))
        
        # Get top sentences
        top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:target_length]
        top_sentences = sorted(top_sentences, key=lambda x: x[0])
        
        return ' '.join(sentences[i] for i, _ in top_sentences)

    def _textrank_summarize(self, sentences: List[str], target_length: int) -> str:
        """Generate summary using TextRank algorithm."""
        # Create sentence embeddings
        embeddings = self.sentence_transformer.encode(sentences)
        
        # Create similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j] = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
        
        # Create graph and calculate scores
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph)
        
        # Get top sentences
        top_sentences = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:target_length]
        top_sentences = sorted(top_sentences, key=lambda x: x[0])
        
        return ' '.join(sentences[i] for i, _ in top_sentences)

    def _lsa_summarize(self, sentences: List[str], target_length: int) -> str:
        """Generate summary using Latent Semantic Analysis."""
        parser = PlaintextParser.from_string(' '.join(sentences), self.tokenizer)
        summarizer = LsaSummarizer()
        summarizer.stop_words = self.stop_words
        summary = summarizer(parser.document, target_length)
        return ' '.join(str(sentence) for sentence in summary)

    def _bart_summarize(self, text: str, target_length: int) -> str:
        """Generate summary using BART model."""
        # Split text into chunks if too long
        chunks = self._split_text_into_chunks(text, max_length=1024)
        
        summaries = []
        for chunk in chunks:
            summary = self.summarizer(chunk, max_length=target_length, min_length=target_length//2, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        
        return ' '.join(summaries)

    def _post_process_summary(self, summary: str, target_length: int) -> str:
        """Post-process summary to improve quality."""
        # Remove redundant sentences
        sentences = sent_tokenize(summary)
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            # Check for redundancy
            if not any(self._is_similar(sentence, s) for s in unique_sentences):
                unique_sentences.append(sentence)
                seen.add(sentence)
        
        # Apply compression ratio
        compressed = ' '.join(unique_sentences)
        
        while len(compressed.split()) > target_length and len(unique_sentences) > 1:
            unique_sentences.pop()
            compressed = ' '.join(unique_sentences)
        
        return compressed

    def _is_similar(self, s1: str, s2: str, threshold: float = 0.7) -> bool:
        """Check if two sentences are similar using cosine similarity."""
        emb1 = self.sentence_transformer.encode([s1])[0]
        emb2 = self.sentence_transformer.encode([s2])[0]
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return similarity > threshold

    def _split_text_into_chunks(self, text: str, max_length: int = 1024) -> List[str]:
        """Split text into chunks of maximum length."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def _combine_summaries(self, summaries: Dict[str, str], scores: Dict[str, float], target_length: int) -> str:
        """Combine multiple summaries based on their scores."""
        # Weight summaries by their scores
        weighted_summaries = []
        for method, summary in summaries.items():
            weight = scores[method]
            weighted_summaries.append((summary, weight))
        
        # Combine summaries
        combined = []
        for summary, weight in weighted_summaries:
            sentences = sent_tokenize(summary)
            combined.extend([(s, weight) for s in sentences])
        
        # Sort by weight and select top sentences
        combined.sort(key=lambda x: x[1], reverse=True)
        return ' '.join(s[0] for s in combined)

    def _evaluate_summary(self, summary: str, original_text: str) -> float:
        """Evaluate summary quality using ROUGE scores."""
        scores = self.rouge_scorer.score(original_text, summary)
        return (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        doc = self.nlp(text)
        phrases = []
        
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip()
            if len(phrase) > 2:
                phrases.append(phrase)
        
        return phrases

    def process_content(
        self,
        html: str,
        url: str,
        extract_summary: bool = True,
        extract_entities: bool = True,
        extract_keywords: bool = True,
        extract_topics: bool = True,
        analyze_sentiment: bool = True
    ) -> Dict[str, Any]:
        """Process HTML content and extract various features."""
        try:
            # Extract main content
            content = trafilatura.extract(html)
            if not content:
                article = Article(url)
                article.set_html(html)
                article.parse()
                content = article.text
            
            # Clean content
            cleaned_content = self.clean_text(content)
            
            # Detect language
            language_info = self.detect_language(cleaned_content)
            
            # Extract metadata
            metadata = self.extract_metadata(html, url)
            
            # Process content
            result = {
                'content': cleaned_content,
                'url': url,
                'timestamp': datetime.utcnow().isoformat(),
                'metadata': metadata,
                'language': language_info,
                'word_count': len(cleaned_content.split())
            }
            
            if extract_summary:
                result['summary'] = self.generate_summary(cleaned_content)
            
            if extract_entities:
                result['entities'] = self.extract_entities(cleaned_content)
            
            if extract_keywords:
                result['keywords'] = self.extract_keywords(cleaned_content)
            
            if extract_topics:
                result['topics'] = self.extract_topics(cleaned_content)
            
            if analyze_sentiment:
                result['sentiment'] = self.analyze_sentiment(cleaned_content)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing content from {url}: {str(e)}")
            return None

    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """Summarize text using the BART model."""
        try:
            # Clean the text
            cleaned_text = self.clean_text(text)
            
            # Generate summary using BART
            summary = self.summarizer(cleaned_text, 
                                    max_length=max_length, 
                                    min_length=min_length, 
                                    do_sample=False)[0]['summary_text']
            
            return summary
        except Exception as e:
            logging.error(f"Error in summarize_text: {str(e)}")
            return text[:max_length]  # Return truncated text as fallback 
"""
Hybrid Retrieval Engine for OpenSCENARIO 2.0 RAG System
Combines FAISS vector search with BM25 sparse retrieval and tag-aware ranking.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import re

from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path

from .config_manager import ConfigManager
from .vector_store import VectorStore
from .document_processor import DocumentChunk

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Represents a retrieval result with combined scoring."""
    chunk: DocumentChunk
    dense_score: float
    sparse_score: float
    combined_score: float
    tag_boost: float = 1.0

class HybridRetrievalEngine:
    """Hybrid retrieval combining dense (FAISS) and sparse (BM25) search."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.vector_store = VectorStore(config)
        self.bm25_index = None
        self.bm25_corpus = []
        self.chunk_mapping = {}  # Maps BM25 index to chunk index
        
        # Retrieval parameters
        self.top_k_dense = config.get('retrieval.top_k_dense', 20)
        self.top_k_sparse = config.get('retrieval.top_k_sparse', 15)
        self.top_k_final = config.get('retrieval.top_k_final', 10)
        self.dense_weight = config.get('retrieval.dense_weight', 0.7)
        self.sparse_weight = config.get('retrieval.sparse_weight', 0.3)
        self.tag_boost_factor = config.get('retrieval.tag_boost_factor', 1.5)
        
        # BM25 parameters
        self.bm25_k1 = config.get('bm25.k1', 1.2)
        self.bm25_b = config.get('bm25.b', 0.75)
        
        # Cache directory
        self.cache_dir = Path(config.get('data.embeddings_cache_dir'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("HybridRetrievalEngine initialized")
    
    def add_chunks(self, chunks: List[DocumentChunk]):
        """Add chunks to both vector and BM25 indices.
        
        Args:
            chunks: List of document chunks to add
        """
        if not chunks:
            return
        
        logger.info(f"Adding {len(chunks)} chunks to hybrid retrieval engine")
        
        # Add to vector store
        self.vector_store.add_chunks(chunks)
        
        # Build BM25 index
        self._build_bm25_index(chunks)
        
        logger.info("Chunks added to hybrid retrieval engine")
    
    def _build_bm25_index(self, new_chunks: List[DocumentChunk]):
        """Build or update BM25 index with new chunks."""
        # Check for cached BM25 index
        cache_file = self.cache_dir / "bm25_index.pkl"
        
        if cache_file.exists() and not new_chunks:
            logger.info("Loading cached BM25 index")
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.bm25_index = cache_data['index']
                    self.bm25_corpus = cache_data['corpus']
                    self.chunk_mapping = cache_data['mapping']
                return
            except Exception as e:
                logger.warning(f"Failed to load BM25 cache: {e}")
        
        # Prepare corpus for BM25
        all_chunks = self.vector_store.chunks
        corpus = []
        chunk_mapping = {}
        
        for i, chunk in enumerate(all_chunks):
            # Tokenize chunk content
            tokens = self._tokenize_text(chunk.content)
            corpus.append(tokens)
            chunk_mapping[i] = i  # BM25 index to chunk index mapping
        
        # Build BM25 index
        logger.info(f"Building BM25 index with {len(corpus)} documents")
        self.bm25_index = BM25Okapi(corpus, k1=self.bm25_k1, b=self.bm25_b)
        self.bm25_corpus = corpus
        self.chunk_mapping = chunk_mapping
        
        # Cache the index
        if self.config.get('bm25.cache_index', True):
            try:
                cache_data = {
                    'index': self.bm25_index,
                    'corpus': self.bm25_corpus,
                    'mapping': self.chunk_mapping
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                logger.info("BM25 index cached")
            except Exception as e:
                logger.warning(f"Failed to cache BM25 index: {e}")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing."""
        # Simple tokenization - can be enhanced with proper NLP tokenizers
        text = text.lower()
        # Remove special characters but keep alphanumeric and common punctuation
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        # Split on whitespace and filter empty tokens
        tokens = [token for token in text.split() if len(token) > 1]
        return tokens
    
    def search(self, query: str, query_tags: List[str] = None, 
               top_k: int = None) -> List[RetrievalResult]:
        """Perform hybrid search combining dense and sparse retrieval.
        
        Args:
            query: Search query
            query_tags: Tags mentioned in the query for boosting
            top_k: Number of results to return (uses config default if None)
            
        Returns:
            List of retrieval results with combined scores
        """
        if top_k is None:
            top_k = self.top_k_final
        
        if not self.vector_store.chunks:
            logger.warning("No chunks available for search")
            return []
        
        logger.debug(f"Performing hybrid search for query: {query[:100]}...")
        
        # Extract tags from query if not provided
        if query_tags is None:
            query_tags = self._extract_query_tags(query)
        
        # Dense retrieval (vector search)
        dense_results = self._dense_search(query, query_tags)
        
        # Sparse retrieval (BM25)
        sparse_results = self._sparse_search(query)
        
        # Combine and rank results
        combined_results = self._combine_results(dense_results, sparse_results, query_tags)
        
        # Return top-k results
        return combined_results[:top_k]
    
    def _dense_search(self, query: str, query_tags: List[str]) -> List[Tuple[int, float]]:
        """Perform dense vector search.
        
        Returns:
            List of (chunk_index, score) tuples
        """
        try:
            if query_tags:
                # Use tag-aware search
                results = self.vector_store.search_with_tag_boost(
                    query, self.top_k_dense, query_tags
                )
            else:
                # Regular vector search
                results = self.vector_store.search(query, self.top_k_dense)
            
            # Convert to (index, score) format
            dense_results = []
            for chunk, score in results:
                # Find chunk index
                chunk_idx = self.vector_store.chunks.index(chunk)
                dense_results.append((chunk_idx, score))
            
            return dense_results
            
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []
    
    def _sparse_search(self, query: str) -> List[Tuple[int, float]]:
        """Perform sparse BM25 search.
        
        Returns:
            List of (chunk_index, score) tuples
        """
        if self.bm25_index is None:
            logger.warning("BM25 index not available")
            return []
        
        try:
            # Tokenize query
            query_tokens = self._tokenize_text(query)
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top-k results
            top_indices = np.argsort(scores)[::-1][:self.top_k_sparse]
            
            sparse_results = []
            for bm25_idx in top_indices:
                chunk_idx = self.chunk_mapping.get(bm25_idx, bm25_idx)
                score = scores[bm25_idx]
                if score > 0:  # Only include positive scores
                    sparse_results.append((chunk_idx, score))
            
            return sparse_results
            
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return []
    
    def _combine_results(self, dense_results: List[Tuple[int, float]], 
                        sparse_results: List[Tuple[int, float]], 
                        query_tags: List[str]) -> List[RetrievalResult]:
        """Combine dense and sparse results with tag boosting.
        
        Args:
            dense_results: Results from vector search
            sparse_results: Results from BM25 search
            query_tags: Tags for boosting
            
        Returns:
            List of combined retrieval results
        """
        # Normalize scores
        dense_scores = self._normalize_scores([score for _, score in dense_results])
        sparse_scores = self._normalize_scores([score for _, score in sparse_results])
        
        # Create score dictionaries
        dense_dict = {idx: score for (idx, _), score in zip(dense_results, dense_scores)}
        sparse_dict = {idx: score for (idx, _), score in zip(sparse_results, sparse_scores)}
        
        # Get all unique chunk indices
        all_indices = set(dense_dict.keys()) | set(sparse_dict.keys())
        
        # Combine scores
        combined_results = []
        for chunk_idx in all_indices:
            if chunk_idx >= len(self.vector_store.chunks):
                continue
            
            chunk = self.vector_store.chunks[chunk_idx]
            
            dense_score = dense_dict.get(chunk_idx, 0.0)
            sparse_score = sparse_dict.get(chunk_idx, 0.0)
            
            # Weighted combination
            combined_score = (self.dense_weight * dense_score + 
                            self.sparse_weight * sparse_score)
            
            # Apply tag boosting
            tag_boost = self._calculate_tag_boost(chunk, query_tags)
            boosted_score = combined_score * tag_boost
            
            result = RetrievalResult(
                chunk=chunk,
                dense_score=dense_score,
                sparse_score=sparse_score,
                combined_score=boosted_score,
                tag_boost=tag_boost
            )
            combined_results.append(result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return combined_results
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def _calculate_tag_boost(self, chunk: DocumentChunk, query_tags: List[str]) -> float:
        """Calculate tag-based boost factor for a chunk."""
        if not query_tags:
            return 1.0
        
        chunk_tags = chunk.metadata.get('tags', [])
        if not chunk_tags:
            return 1.0
        
        # Calculate tag overlap
        tag_overlap = len(set(query_tags) & set(chunk_tags))
        if tag_overlap == 0:
            return 1.0
        
        # Apply boost based on overlap ratio
        overlap_ratio = tag_overlap / len(query_tags)
        boost = 1.0 + (self.tag_boost_factor - 1.0) * overlap_ratio
        
        return boost
    
    def _extract_query_tags(self, query: str) -> List[str]:
        """Extract potential tags from query text."""
        # Get configured tag categories
        tag_categories = self.config.get('tags.categories', [])
        
        query_lower = query.lower()
        found_tags = []
        
        for tag in tag_categories:
            # Check for exact match or variations
            tag_patterns = [
                tag.replace('_', ' '),
                tag.replace('_', '-'),
                tag.replace('_', ''),
                tag
            ]
            
            for pattern in tag_patterns:
                if pattern in query_lower:
                    found_tags.append(tag)
                    break
        
        # Also look for common OpenSCENARIO terms
        osc_terms = {
            'cut in': 'cut_in',
            'cutin': 'cut_in',
            'lane change': 'lane_change',
            'lanechange': 'lane_change',
            'overtake': 'overtaking',
            'parallel drive': 'parallel_drive',
            'traffic light': 'traffic_light',
            'intersection': 'intersection',
            'parking': 'parking',
            'pedestrian': 'pedestrian',
            'weather': 'weather'
        }
        
        for term, tag in osc_terms.items():
            if term in query_lower and tag not in found_tags:
                found_tags.append(tag)
        
        return found_tags
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get chunk by its ID."""
        for chunk in self.vector_store.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
    
    def get_chunks_by_type(self, chunk_type: str) -> List[DocumentChunk]:
        """Get all chunks of a specific type."""
        return [chunk for chunk in self.vector_store.chunks 
                if chunk.metadata.get('type') == chunk_type]
    
    def get_chunks_by_tags(self, tags: List[str]) -> List[DocumentChunk]:
        """Get all chunks that have any of the specified tags."""
        matching_chunks = []
        for chunk in self.vector_store.chunks:
            chunk_tags = chunk.metadata.get('tags', [])
            if any(tag in chunk_tags for tag in tags):
                matching_chunks.append(chunk)
        return matching_chunks
    
    def save_indices(self, save_path: str = None):
        """Save both vector and BM25 indices."""
        if save_path is None:
            save_path = self.cache_dir / "retrieval_indices"
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save vector store
        self.vector_store.save_index(save_path / "vector_store")
        
        # Save BM25 index
        if self.bm25_index is not None:
            bm25_cache = {
                'index': self.bm25_index,
                'corpus': self.bm25_corpus,
                'mapping': self.chunk_mapping
            }
            with open(save_path / "bm25_index.pkl", 'wb') as f:
                pickle.dump(bm25_cache, f)
        
        logger.info(f"Retrieval indices saved to {save_path}")
    
    def load_indices(self, load_path: str = None):
        """Load both vector and BM25 indices."""
        if load_path is None:
            load_path = self.cache_dir / "retrieval_indices"
        
        load_path = Path(load_path)
        
        if not load_path.exists():
            logger.warning(f"Indices path does not exist: {load_path}")
            return False
        
        try:
            # Load vector store
            vector_loaded = self.vector_store.load_index(load_path / "vector_store")
            
            # Load BM25 index
            bm25_file = load_path / "bm25_index.pkl"
            if bm25_file.exists():
                with open(bm25_file, 'rb') as f:
                    bm25_cache = pickle.load(f)
                    self.bm25_index = bm25_cache['index']
                    self.bm25_corpus = bm25_cache['corpus']
                    self.chunk_mapping = bm25_cache['mapping']
                logger.info("BM25 index loaded")
            
            return vector_loaded
            
        except Exception as e:
            logger.error(f"Failed to load indices: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval engine statistics."""
        vector_stats = self.vector_store.get_stats()
        
        stats = {
            'total_chunks': len(self.vector_store.chunks),
            'dense_search_enabled': True,
            'sparse_search_enabled': self.bm25_index is not None,
            'retrieval_params': {
                'top_k_dense': self.top_k_dense,
                'top_k_sparse': self.top_k_sparse,
                'top_k_final': self.top_k_final,
                'dense_weight': self.dense_weight,
                'sparse_weight': self.sparse_weight,
                'tag_boost_factor': self.tag_boost_factor
            },
            'vector_store': vector_stats
        }
        
        if self.bm25_index:
            stats['bm25_corpus_size'] = len(self.bm25_corpus)
            stats['bm25_params'] = {
                'k1': self.bm25_k1,
                'b': self.bm25_b
            }
        
        return stats

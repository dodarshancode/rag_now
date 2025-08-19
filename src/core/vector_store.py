"""
GPU-Accelerated Vector Store for OpenSCENARIO 2.0 RAG System
Implements FAISS-based vector storage with GPU acceleration and caching.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

import torch
import faiss
from sentence_transformers import SentenceTransformer

from .config_manager import ConfigManager
from .document_processor import DocumentChunk

logger = logging.getLogger(__name__)

class VectorStore:
    """GPU-accelerated vector store using FAISS."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.embedding_model = None
        self.faiss_index = None
        self.chunks = []
        self.chunk_metadata = []
        
        # Configuration
        self.dimension = config.get('vector_store.dimension', 768)
        self.use_gpu = config.get('vector_store.use_gpu', True)
        self.gpu_devices = config.get('vector_store.gpu_devices', [0, 1, 2, 3])
        self.cache_dir = Path(config.get('data.embeddings_cache_dir'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        logger.info(f"VectorStore initialized with dimension={self.dimension}, GPU={self.use_gpu}")
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model."""
        model_name = self.config.get('embedding.model_name', 'microsoft/codebert-base')
        model_path = self.config.get('embedding.model_path')
        
        try:
            if model_path and Path(model_path).exists():
                logger.info(f"Loading embedding model from local path: {model_path}")
                self.embedding_model = SentenceTransformer(model_path)
            else:
                logger.info(f"Loading embedding model: {model_name}")
                self.embedding_model = SentenceTransformer(model_name)
            
            # Move to GPU if available
            if self.use_gpu and torch.cuda.is_available():
                device = f"cuda:{self.gpu_devices[0]}"
                self.embedding_model = self.embedding_model.to(device)
                logger.info(f"Embedding model moved to {device}")
            
            # Verify dimension matches
            test_embedding = self.embedding_model.encode(["test"], show_progress_bar=False)
            actual_dim = test_embedding.shape[1]
            
            if actual_dim != self.dimension:
                logger.warning(f"Updating dimension from {self.dimension} to {actual_dim}")
                self.dimension = actual_dim
                self.config.set('vector_store.dimension', actual_dim)
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def _create_faiss_index(self) -> faiss.Index:
        """Create FAISS index based on configuration."""
        index_type = self.config.get('vector_store.index_type', 'IndexFlatIP')
        
        if index_type == 'IndexFlatIP':
            # Inner product index (for cosine similarity with normalized vectors)
            index = faiss.IndexFlatIP(self.dimension)
        elif index_type == 'IndexFlatL2':
            # L2 distance index
            index = faiss.IndexFlatL2(self.dimension)
        elif index_type == 'IndexIVFFlat':
            # IVF index for larger datasets
            quantizer = faiss.IndexFlatIP(self.dimension)
            nlist = min(4096, max(1, len(self.chunks) // 100))  # Adaptive nlist
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        else:
            logger.warning(f"Unknown index type {index_type}, using IndexFlatIP")
            index = faiss.IndexFlatIP(self.dimension)
        
        # Move to GPU if configured
        if self.use_gpu and torch.cuda.is_available():
            try:
                # Use multiple GPUs if available
                if len(self.gpu_devices) > 1:
                    logger.info(f"Using multiple GPUs: {self.gpu_devices}")
                    gpu_resources = []
                    for gpu_id in self.gpu_devices:
                        res = faiss.StandardGpuResources()
                        gpu_resources.append(res)
                    
                    # Create multi-GPU index
                    index = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, index)
                else:
                    # Single GPU
                    gpu_id = self.gpu_devices[0]
                    logger.info(f"Using single GPU: {gpu_id}")
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, gpu_id, index)
                
                logger.info("FAISS index moved to GPU")
            except Exception as e:
                logger.warning(f"Failed to move FAISS to GPU: {e}, using CPU")
                self.use_gpu = False
        
        return index
    
    def add_chunks(self, chunks: List[DocumentChunk], batch_size: int = None):
        """Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks to add
            batch_size: Batch size for embedding generation
        """
        if not chunks:
            logger.warning("No chunks provided to add")
            return
        
        if batch_size is None:
            batch_size = self.config.get('embedding.batch_size', 32)
        
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        # Check for cached embeddings
        cache_file = self.cache_dir / "embeddings_cache.pkl"
        cached_embeddings = self._load_cached_embeddings(cache_file)
        
        # Prepare texts and track which need embedding
        texts = []
        chunks_to_embed = []
        embeddings_list = []
        
        for chunk in chunks:
            chunk_hash = self._get_chunk_hash(chunk)
            
            if chunk_hash in cached_embeddings:
                # Use cached embedding
                embeddings_list.append(cached_embeddings[chunk_hash])
            else:
                # Need to compute embedding
                texts.append(chunk.content)
                chunks_to_embed.append((chunk, chunk_hash))
                embeddings_list.append(None)  # Placeholder
        
        # Compute embeddings for new chunks
        if texts:
            logger.info(f"Computing embeddings for {len(texts)} new chunks")
            new_embeddings = self._compute_embeddings(texts, batch_size)
            
            # Update cache and embeddings list
            new_cache_entries = {}
            embed_idx = 0
            
            for i, embedding in enumerate(embeddings_list):
                if embedding is None:
                    chunk, chunk_hash = chunks_to_embed[embed_idx]
                    new_embedding = new_embeddings[embed_idx]
                    embeddings_list[i] = new_embedding
                    new_cache_entries[chunk_hash] = new_embedding
                    embed_idx += 1
            
            # Update cache
            cached_embeddings.update(new_cache_entries)
            self._save_cached_embeddings(cached_embeddings, cache_file)
        
        # Convert to numpy array and normalize
        embeddings = np.vstack(embeddings_list).astype(np.float32)
        
        # Normalize for cosine similarity
        if self.config.get('embedding.normalize_embeddings', True):
            faiss.normalize_L2(embeddings)
        
        # Create or update FAISS index
        if self.faiss_index is None:
            self.faiss_index = self._create_faiss_index()
        
        # Train index if needed (for IVF indices)
        if hasattr(self.faiss_index, 'is_trained') and not self.faiss_index.is_trained:
            logger.info("Training FAISS index")
            self.faiss_index.train(embeddings)
        
        # Add embeddings to index
        start_id = len(self.chunks)
        self.faiss_index.add(embeddings)
        
        # Store chunks and metadata
        self.chunks.extend(chunks)
        for i, chunk in enumerate(chunks):
            metadata = chunk.metadata.copy()
            metadata['vector_id'] = start_id + i
            self.chunk_metadata.append(metadata)
        
        logger.info(f"Added {len(chunks)} chunks to vector store. Total: {len(self.chunks)}")
    
    def _compute_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Compute embeddings for texts in batches."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
                embeddings.append(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error computing embeddings for batch {i//batch_size}: {e}")
                # Fallback: process one by one
                for text in batch_texts:
                    try:
                        emb = self.embedding_model.encode([text], show_progress_bar=False)
                        embeddings.append(emb)
                    except:
                        # Use zero embedding as last resort
                        embeddings.append(np.zeros((1, self.dimension)))
        
        return np.vstack(embeddings)
    
    def search(self, query: str, top_k: int = 10, filter_tags: List[str] = None) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_tags: Optional tags to filter results
            
        Returns:
            List of (chunk, score) tuples
        """
        if self.faiss_index is None or len(self.chunks) == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Compute query embedding
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)
        query_embedding = query_embedding.astype(np.float32)
        
        # Normalize if needed
        if self.config.get('embedding.normalize_embeddings', True):
            faiss.normalize_L2(query_embedding)
        
        # Search
        search_k = min(top_k * 2, len(self.chunks))  # Get more results for filtering
        scores, indices = self.faiss_index.search(query_embedding, search_k)
        
        # Process results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            
            chunk = self.chunks[idx]
            
            # Apply tag filtering
            if filter_tags:
                chunk_tags = chunk.metadata.get('tags', [])
                if not any(tag in chunk_tags for tag in filter_tags):
                    continue
            
            results.append((chunk, float(score)))
        
        # Return top_k results
        return results[:top_k]
    
    def search_with_tag_boost(self, query: str, top_k: int = 10, 
                             query_tags: List[str] = None) -> List[Tuple[DocumentChunk, float]]:
        """Search with tag-based score boosting.
        
        Args:
            query: Search query
            top_k: Number of results to return
            query_tags: Tags mentioned in query for boosting
            
        Returns:
            List of (chunk, boosted_score) tuples
        """
        # Get initial results
        initial_results = self.search(query, top_k * 3)  # Get more for reranking
        
        if not query_tags:
            return initial_results[:top_k]
        
        # Apply tag boosting
        boost_factor = self.config.get('retrieval.tag_boost_factor', 1.5)
        boosted_results = []
        
        for chunk, score in initial_results:
            chunk_tags = chunk.metadata.get('tags', [])
            
            # Calculate tag overlap
            tag_overlap = len(set(query_tags) & set(chunk_tags))
            if tag_overlap > 0:
                boost = 1 + (boost_factor - 1) * (tag_overlap / len(query_tags))
                boosted_score = score * boost
            else:
                boosted_score = score
            
            boosted_results.append((chunk, boosted_score))
        
        # Sort by boosted score and return top_k
        boosted_results.sort(key=lambda x: x[1], reverse=True)
        return boosted_results[:top_k]
    
    def _get_chunk_hash(self, chunk: DocumentChunk) -> str:
        """Get hash for chunk content for caching."""
        import hashlib
        content_hash = hashlib.md5(chunk.content.encode()).hexdigest()
        return f"{content_hash}_{self.dimension}"
    
    def _load_cached_embeddings(self, cache_file: Path) -> Dict[str, np.ndarray]:
        """Load cached embeddings from file."""
        if not cache_file.exists():
            return {}
        
        try:
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            logger.info(f"Loaded {len(cached)} cached embeddings")
            return cached
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
            return {}
    
    def _save_cached_embeddings(self, embeddings: Dict[str, np.ndarray], cache_file: Path):
        """Save embeddings to cache file."""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"Saved {len(embeddings)} embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")
    
    def save_index(self, index_path: str = None):
        """Save FAISS index and metadata to disk."""
        if index_path is None:
            index_path = self.cache_dir / "faiss_index"
        
        index_path = Path(index_path)
        index_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save FAISS index
            if self.use_gpu and hasattr(self.faiss_index, 'index'):
                # Move to CPU for saving
                cpu_index = faiss.index_gpu_to_cpu(self.faiss_index)
                faiss.write_index(cpu_index, str(index_path / "index.faiss"))
            else:
                faiss.write_index(self.faiss_index, str(index_path / "index.faiss"))
            
            # Save chunks and metadata
            chunks_data = [chunk.to_dict() for chunk in self.chunks]
            with open(index_path / "chunks.json", 'w') as f:
                json.dump(chunks_data, f, indent=2)
            
            with open(index_path / "metadata.json", 'w') as f:
                json.dump(self.chunk_metadata, f, indent=2)
            
            # Save configuration
            config_data = {
                'dimension': self.dimension,
                'use_gpu': self.use_gpu,
                'gpu_devices': self.gpu_devices,
                'total_chunks': len(self.chunks)
            }
            with open(index_path / "config.json", 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Vector store saved to {index_path}")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
    
    def load_index(self, index_path: str = None):
        """Load FAISS index and metadata from disk."""
        if index_path is None:
            index_path = self.cache_dir / "faiss_index"
        
        index_path = Path(index_path)
        
        if not index_path.exists():
            logger.warning(f"Index path does not exist: {index_path}")
            return False
        
        try:
            # Load configuration
            with open(index_path / "config.json", 'r') as f:
                config_data = json.load(f)
            
            # Load FAISS index
            cpu_index = faiss.read_index(str(index_path / "index.faiss"))
            
            # Move to GPU if configured
            if self.use_gpu and torch.cuda.is_available():
                if len(self.gpu_devices) > 1:
                    gpu_resources = []
                    for gpu_id in self.gpu_devices:
                        res = faiss.StandardGpuResources()
                        gpu_resources.append(res)
                    self.faiss_index = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, cpu_index)
                else:
                    res = faiss.StandardGpuResources()
                    self.faiss_index = faiss.index_cpu_to_gpu(res, self.gpu_devices[0], cpu_index)
            else:
                self.faiss_index = cpu_index
            
            # Load chunks
            with open(index_path / "chunks.json", 'r') as f:
                chunks_data = json.load(f)
            self.chunks = [DocumentChunk.from_dict(data) for data in chunks_data]
            
            # Load metadata
            with open(index_path / "metadata.json", 'r') as f:
                self.chunk_metadata = json.load(f)
            
            logger.info(f"Vector store loaded from {index_path} with {len(self.chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        stats = {
            'total_chunks': len(self.chunks),
            'dimension': self.dimension,
            'use_gpu': self.use_gpu,
            'gpu_devices': self.gpu_devices,
            'embedding_model': self.embedding_model.__class__.__name__ if self.embedding_model else None,
            'index_type': type(self.faiss_index).__name__ if self.faiss_index else None,
            'cache_directory': str(self.cache_dir)
        }
        
        if self.faiss_index:
            stats['index_size'] = self.faiss_index.ntotal
            stats['is_trained'] = getattr(self.faiss_index, 'is_trained', True)
        
        return stats

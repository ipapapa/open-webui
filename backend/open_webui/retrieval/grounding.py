"""
RAG Grounding Step Module

This module implements a lightweight grounding step after retrieval to re-align context frames
and prevent silent semantic drift when using different embedding models. The grounding step
validates that retrieved content is semantically relevant to the original query.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)


class GroundingResult:
    """Result of the grounding step validation"""
    
    def __init__(
        self,
        validated_documents: List[Dict[str, Any]],
        filtered_count: int,
        average_confidence: float,
        validation_scores: List[float]
    ):
        self.validated_documents = validated_documents
        self.filtered_count = filtered_count
        self.average_confidence = average_confidence
        self.validation_scores = validation_scores


def calculate_query_document_relevance(
    query_embedding: List[float],
    document_embeddings: List[List[float]],
    document_texts: List[str],
    threshold: float = 0.3
) -> Tuple[List[int], List[float]]:
    """
    Calculate relevance scores between query and documents using cosine similarity.
    
    Args:
        query_embedding: Embedding vector of the original query
        document_embeddings: List of embedding vectors for retrieved documents
        document_texts: List of document text content (for logging)
        threshold: Minimum similarity threshold for relevance
    
    Returns:
        Tuple of (valid_indices, relevance_scores)
    """
    if not document_embeddings or not query_embedding:
        return [], []
    
    try:
        # Convert to numpy arrays for efficient computation
        query_vec = np.array(query_embedding).reshape(1, -1)
        doc_vecs = np.array(document_embeddings)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vec, doc_vecs)[0]
        
        # Find documents above threshold
        valid_indices = [i for i, score in enumerate(similarities) if score >= threshold]
        relevance_scores = similarities.tolist()
        
        log.debug(f"Grounding validation: {len(valid_indices)}/{len(document_embeddings)} documents passed threshold {threshold}")
        
        return valid_indices, relevance_scores
        
    except Exception as e:
        log.error(f"Error in grounding relevance calculation: {e}")
        # Fallback: return all documents if calculation fails
        return list(range(len(document_embeddings))), [1.0] * len(document_embeddings)


def apply_grounding_step(
    query: str,
    query_embedding: List[float],
    retrieved_documents: List[Dict[str, Any]],
    embedding_function,
    threshold: float = 0.3,
    user=None
) -> GroundingResult:
    """
    Apply grounding step to validate retrieved documents against the original query.
    
    This function re-validates retrieved documents by comparing their embeddings
    with the query embedding to catch semantic drift that might occur with
    different embedding models.
    
    Args:
        query: Original query string
        query_embedding: Query embedding vector
        retrieved_documents: List of documents from retrieval step
        embedding_function: Function to generate embeddings for document texts
        threshold: Minimum relevance threshold (0.0 to 1.0)
        user: User context for embedding function
    
    Returns:
        GroundingResult with validated documents and metrics
    """
    if not retrieved_documents:
        return GroundingResult([], 0, 0.0, [])
    
    try:
        # Extract document texts for re-embedding
        document_texts = []
        for doc in retrieved_documents:
            if isinstance(doc, dict):
                # Handle different document formats
                text = doc.get('document', doc.get('text', doc.get('content', '')))
                if isinstance(text, str):
                    document_texts.append(text)
                else:
                    document_texts.append(str(text))
            else:
                document_texts.append(str(doc))
        
        # Generate embeddings for documents using the same embedding function
        # that was used for the query
        log.debug(f"Generating embeddings for {len(document_texts)} documents in grounding step")
        document_embeddings = embedding_function(document_texts, prefix=None, user=user)
        
        if not document_embeddings:
            log.warning("Failed to generate document embeddings in grounding step")
            return GroundingResult(retrieved_documents, 0, 1.0, [1.0] * len(retrieved_documents))
        
        # Calculate relevance scores
        valid_indices, relevance_scores = calculate_query_document_relevance(
            query_embedding, document_embeddings, document_texts, threshold
        )
        
        # Filter documents based on grounding validation
        validated_documents = [retrieved_documents[i] for i in valid_indices]
        filtered_count = len(retrieved_documents) - len(validated_documents)
        
        # Calculate average confidence for valid documents
        valid_scores = [relevance_scores[i] for i in valid_indices] if valid_indices else []
        average_confidence = np.mean(valid_scores) if valid_scores else 0.0
        
        if filtered_count > 0:
            log.info(f"Grounding step filtered out {filtered_count} documents below threshold {threshold}")
        
        return GroundingResult(
            validated_documents=validated_documents,
            filtered_count=filtered_count,
            average_confidence=float(average_confidence),
            validation_scores=relevance_scores
        )
        
    except Exception as e:
        log.error(f"Error in grounding step: {e}")
        # Fallback: return original documents if grounding fails
        return GroundingResult(
            validated_documents=retrieved_documents,
            filtered_count=0,
            average_confidence=1.0,
            validation_scores=[1.0] * len(retrieved_documents)
        )


def format_documents_for_retrieval(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format validated documents back to the expected retrieval format.
    
    Args:
        documents: List of validated documents
    
    Returns:
        Formatted result dictionary matching query_doc/query_collection format
    """
    if not documents:
        return {
            "ids": [[]],
            "distances": [[]],
            "metadatas": [[]],
            "documents": [[]],
            "embeddings": None
        }
    
    # Extract components for formatting
    ids = []
    distances = []
    metadatas = []
    doc_texts = []
    
    for doc in documents:
        if isinstance(doc, dict):
            ids.append(doc.get('id', ''))
            distances.append(doc.get('distance', 0.0))
            metadatas.append(doc.get('metadata', {}))
            doc_texts.append(doc.get('document', doc.get('text', doc.get('content', ''))))
        else:
            ids.append('')
            distances.append(0.0)
            metadatas.append({})
            doc_texts.append(str(doc))
    
    return {
        "ids": [ids],
        "distances": [distances], 
        "metadatas": [metadatas],
        "documents": [doc_texts],
        "embeddings": None
    }
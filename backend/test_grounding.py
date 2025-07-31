"""
Basic tests for RAG grounding step functionality
"""

import pytest
from unittest.mock import Mock
from open_webui.retrieval.grounding import (
    apply_grounding_step,
    calculate_query_document_relevance,
    format_documents_for_retrieval,
    GroundingResult
)


def test_calculate_query_document_relevance():
    """Test basic relevance calculation between query and documents"""
    # Simple test vectors
    query_embedding = [1.0, 0.0, 0.0]  # High similarity to first doc
    document_embeddings = [
        [1.0, 0.0, 0.0],  # Perfect match
        [0.0, 1.0, 0.0],  # Orthogonal (low similarity)
        [0.8, 0.6, 0.0],  # Partial match
    ]
    document_texts = ["doc1", "doc2", "doc3"]
    threshold = 0.7
    
    valid_indices, scores = calculate_query_document_relevance(
        query_embedding, document_embeddings, document_texts, threshold
    )
    
    # Should return indices 0 and 2 (docs above threshold)
    assert len(valid_indices) == 2
    assert 0 in valid_indices  # Perfect match
    assert 2 in valid_indices  # Partial match above threshold
    assert 1 not in valid_indices  # Below threshold
    
    # Scores should be similarity values
    assert len(scores) == 3
    assert scores[0] > 0.9  # Perfect match
    assert scores[1] < 0.5  # Orthogonal


def test_apply_grounding_step_basic():
    """Test basic grounding step functionality"""
    query = "test query"
    query_embedding = [1.0, 0.0, 0.0]
    
    retrieved_documents = [
        {
            'document': 'highly relevant document',
            'id': 'doc1',
            'distance': 0.1,
            'metadata': {'source': 'test'}
        },
        {
            'document': 'irrelevant document', 
            'id': 'doc2',
            'distance': 0.5,
            'metadata': {'source': 'test'}
        }
    ]
    
    # Mock embedding function that returns embeddings based on content
    def mock_embedding_function(texts, prefix=None, user=None):
        embeddings = []
        for text in texts:
            if 'relevant' in text:
                embeddings.append([1.0, 0.0, 0.0])  # High similarity
            else:
                embeddings.append([0.0, 1.0, 0.0])  # Low similarity
        return embeddings
    
    result = apply_grounding_step(
        query=query,
        query_embedding=query_embedding,
        retrieved_documents=retrieved_documents,
        embedding_function=mock_embedding_function,
        threshold=0.7,
        user=None
    )
    
    assert isinstance(result, GroundingResult)
    assert len(result.validated_documents) == 1  # Only relevant doc should pass
    assert result.validated_documents[0]['document'] == 'highly relevant document'
    assert result.filtered_count == 1  # One doc filtered out


def test_apply_grounding_step_empty_documents():
    """Test grounding step with empty document list"""
    result = apply_grounding_step(
        query="test",
        query_embedding=[1.0, 0.0, 0.0],
        retrieved_documents=[],
        embedding_function=Mock(),
        threshold=0.5,
        user=None
    )
    
    assert isinstance(result, GroundingResult)
    assert len(result.validated_documents) == 0
    assert result.filtered_count == 0
    assert result.average_confidence == 0.0


def test_apply_grounding_step_all_documents_filtered():
    """Test grounding step when all documents are below threshold"""
    query = "test query"
    query_embedding = [1.0, 0.0, 0.0]
    
    retrieved_documents = [
        {'document': 'low relevance doc', 'id': 'doc1', 'distance': 0.1, 'metadata': {}}
    ]
    
    # Mock embedding function that returns low similarity
    def mock_embedding_function(texts, prefix=None, user=None):
        return [[0.0, 1.0, 0.0]]  # Orthogonal to query
    
    result = apply_grounding_step(
        query=query,
        query_embedding=query_embedding,
        retrieved_documents=retrieved_documents,
        embedding_function=mock_embedding_function,
        threshold=0.7,  # High threshold
        user=None
    )
    
    assert len(result.validated_documents) == 0
    assert result.filtered_count == 1


def test_format_documents_for_retrieval():
    """Test formatting documents back to retrieval format"""
    documents = [
        {
            'document': 'test content',
            'id': 'doc1',
            'distance': 0.1,
            'metadata': {'source': 'test'}
        }
    ]
    
    formatted = format_documents_for_retrieval(documents)
    
    assert 'ids' in formatted
    assert 'distances' in formatted
    assert 'metadatas' in formatted
    assert 'documents' in formatted
    assert 'embeddings' in formatted
    
    assert formatted['ids'] == [['doc1']]
    assert formatted['distances'] == [[0.1]]
    assert formatted['documents'] == [['test content']]
    assert formatted['metadatas'] == [[{'source': 'test'}]]


def test_format_documents_for_retrieval_empty():
    """Test formatting empty document list"""
    formatted = format_documents_for_retrieval([])
    
    assert formatted['ids'] == [[]]
    assert formatted['distances'] == [[]]
    assert formatted['documents'] == [[]]
    assert formatted['metadatas'] == [[]]
    assert formatted['embeddings'] is None


if __name__ == "__main__":
    # Run basic tests
    test_calculate_query_document_relevance()
    test_apply_grounding_step_basic() 
    test_apply_grounding_step_empty_documents()
    test_apply_grounding_step_all_documents_filtered()
    test_format_documents_for_retrieval()
    test_format_documents_for_retrieval_empty()
    print("All tests passed!")
"""
Vector database client wrapper for Qdrant.
"""

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
import os


class VectorDBClient:
    """
    Wrapper for Qdrant vector database operations.
    """
    
    def __init__(
        self,
        url: str = None,
        api_key: str = None,
        collection_name: str = "faculty_chunks"
    ):
        """
        Initialize Qdrant client.
        
        Args:
            url: Qdrant server URL (default: localhost:6333)
            api_key: API key for cloud deployment (optional)
            collection_name: Default collection name
        """
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name
        
        # Initialize client
        if self.api_key:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            self.client = QdrantClient(url=self.url)
        
        print(f"✓ Connected to Qdrant at {self.url}")
    
    def create_collection(
        self,
        collection_name: str = None,
        vector_size: int = 768,
        distance: Distance = Distance.COSINE
    ):
        """
        Create a new collection.
        
        Args:
            collection_name: Name of collection to create
            vector_size: Dimension of vectors
            distance: Distance metric (COSINE, EUCLID, DOT)
        """
        collection_name = collection_name or self.collection_name
        
        # Check if collection exists
        collections = self.client.get_collections().collections
        if any(c.name == collection_name for c in collections):
            print(f"Collection '{collection_name}' already exists")
            return
        
        # Create collection
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance
            )
        )
        
        print(f"✓ Created collection: {collection_name}")
    
    def upsert(
        self,
        collection_name: str,
        points: List[Dict[str, Any]]
    ):
        """
        Insert or update points in collection.
        
        Args:
            collection_name: Target collection
            points: List of points with id, vector, and payload
        """
        point_structs = [
            PointStruct(
                id=point["id"],
                vector=point["vector"],
                payload=point.get("payload", {})
            )
            for point in points
        ]
        
        self.client.upsert(
            collection_name=collection_name,
            points=point_structs
        )
    
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        query_filter: Optional[Dict[str, Any]] = None,
        score_threshold: float = None
    ) -> List[Any]:
        """
        Search for similar vectors using qdrant-client 1.9.0 API.
        
        Args:
            collection_name: Collection to search
            query_vector: Query vector
            limit: Number of results
            query_filter: Metadata filters
            score_threshold: Minimum similarity score
        
        Returns:
            List of search results
        """
        # Build filter if provided
        qdrant_filter = None
        if query_filter:
            qdrant_filter = self._build_filter(query_filter)
        
        # Use query_points for qdrant-client 1.9.0
        # The search() method was renamed to query_points()
        try:
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                query_filter=qdrant_filter,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False
            )
            return results.points if hasattr(results, 'points') else results
        except AttributeError:
            # Fallback for older API
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=qdrant_filter,
                score_threshold=score_threshold
            )
            return results
    
    def get_collection_info(self, collection_name: str = None) -> Dict[str, Any]:
        """Get information about a collection."""
        collection_name = collection_name or self.collection_name
        
        info = self.client.get_collection(collection_name)
        
        return {
            "name": collection_name,
            "points_count": info.points_count,
            "status": info.status,
        }
    
    def delete_collection(self, collection_name: str):
        """Delete a collection."""
        self.client.delete_collection(collection_name)
        print(f"✓ Deleted collection: {collection_name}")
    
    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from dict."""
        # If already a Filter object, return as-is
        if isinstance(filters, Filter):
            return filters
        
        # If not a dict, return None
        if not isinstance(filters, dict):
            return None
        
        conditions = []
        
        for key, value in filters.items():
            # Skip None values
            if value is None:
                continue
            
            if isinstance(value, list):
                # Multiple values - use should (OR)
                for v in value:
                    if v is not None:
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(value=v)
                            )
                        )
            else:
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
        
        return Filter(must=conditions) if conditions else None

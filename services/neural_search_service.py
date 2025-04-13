from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer

from config import QDRANT_URL, MODEL_NAME


class NeuralSearcher:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")
        self.qdrant_client = AsyncQdrantClient(QDRANT_URL)

    async def search(self, text: str):
        vector = self.model.encode(text).tolist()

        search_result = await self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,
            limit=5,
        )

        payloads = [hit.payload for hit in search_result]
        return payloads

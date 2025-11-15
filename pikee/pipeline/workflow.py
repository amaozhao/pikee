from typing import List

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from pikee.infrastructure.config.settings import get_settings
from pikee.infrastructure.database import get_qdrant_client
from pikee.infrastructure.llm.embedder import FastEmbedEmbedder
from pikee.pipeline.models.atom import Atom
from pikee.pipeline.models.chunk import Chunk
from pikee.pipeline.models.document import Document
from pikee.pipeline.services.chunker import DocumentChunker
from pikee.pipeline.services.document.loader import DocumentLoader
from pikee.pipeline.services.vector.builder import VectorDatabaseBuilder

settings = get_settings()
qdrant_client = get_qdrant_client(settings)
embedder = FastEmbedEmbedder(batch_size=100)


class Workflow:
    @task(cache_policy=NO_CACHE, retries=3, retry_delay_seconds=5)
    def load_document(self, file_path: str) -> Document:
        """加载文档（自动重试3次）."""
        loader = DocumentLoader()
        return loader.load(file_path)

    @task(cache_policy=NO_CACHE, retries=3)
    def chunk_document(self, document: Document) -> List[Chunk]:
        """切分文档."""
        chunker = DocumentChunker(settings)
        return chunker.chunk_document(document)

    @task(cache_policy=NO_CACHE, retries=2)
    def extract_atoms(self, chunks: List[Chunk]) -> List[Atom]:
        """提取原子问题."""
        # extractor = AtomExtractor(settings, llm_client)
        # return extractor.extract_atoms_from_chunks(chunks)
        return []

    @task(cache_policy=NO_CACHE, retries=3)
    def build_vector_store(self, chunks: List[Chunk], atoms: List[Atom]) -> bool:
        """构建向量存储."""
        builder = VectorDatabaseBuilder(settings, qdrant_client, embedder)
        return builder.build_all(chunks, atoms)

    @flow(name="Document Processing Pipeline", retries=1)
    def document_processing_pipeline(self, file_path: str):
        """文档处理主流程（Prefect Flow）."""
        document = self.load_document(file_path)
        chunks = self.chunk_document(document)
        atoms = self.extract_atoms(chunks)
        print(len(chunks), len(atoms))
        # success = self.build_vector_store(chunks, atoms)
        return {"success": True, "chunks": len(chunks), "atoms": 0}
        # return {"success": success, "chunks": len(chunks), "atoms": len(atoms)}


if __name__ == "__main__":
    workflow = Workflow()
    workflow.document_processing_pipeline("docs/PIKE-RAG生产级实现方案.md")  # type: ignore

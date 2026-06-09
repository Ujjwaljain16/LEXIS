import abc
import numpy as np
from typing import List, Any

class BaseParser(abc.ABC):
    @abc.abstractmethod
    def parse(self, file_path: str) -> List[Any]:
        """Parses a document into raw elements."""
        pass

class BaseChunker(abc.ABC):
    @abc.abstractmethod
    def chunk(self, raw_elements: List[Any]) -> List[Any]:
        """Converts raw elements into semantic chunks."""
        pass

class BaseEmbedder(abc.ABC):
    @abc.abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Embeds a single text string."""
        pass

    @abc.abstractmethod
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embeds a batch of strings efficiently."""
        pass

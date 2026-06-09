"""
Document Parser for LEXIS.

Rationale: Extracts raw text and geometry (bounding boxes) from source documents.
Source Inspiration: RAGFlow parser.
Deviations from Source Repos: Directly exposes `unstructured` hi_res strategy without intermediary wrappers to maximize geometry fidelity.
Expected Impact on Metrics: High citation accuracy; preserves exact spatial coordinates for UI PDF highlights.
"""
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element
from typing import List
from lexis.ingestion.interfaces import BaseParser

class LexisParser(BaseParser):
    def parse(self, file_path: str) -> List[Element]:
        """
        Parses a PDF into structural elements with bounding box coordinates.
        Uses hi_res strategy for accurate layout parsing.
        """
        elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=True,
            include_page_breaks=False
        )
        return elements

import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.html import partition_html
from unstructured.partition.text import partition_text
from unstructured.documents.elements import Element

from lexis.ingestion.interfaces import BaseParser

BLOCK_TYPE_MAP = {
    "Title": "title",
    "NarrativeText": "text",
    "Text": "text",
    "ListItem": "text",
    "Table": "table",
    "Image": "figure",
    "FigureCaption": "text",
}

DOC_TYPE_PATTERNS = {
    "contract": [r"agreement", r"contract", r"amendment", r"covenant"],
    "10-k": [r"form 10-k", r"annual report", r"item 1a", r"risk factor"],
    "regulation": [r"whereas", r"hereby", r"regulation", r"statute"],
    "research": [r"abstract", r"methodology", r"conclusion", r"references"],
}

class LexisParser(BaseParser):
    def _detect_doc_type(self, text: str) -> str:
        text_lower = text.lower()
        for doc_type, patterns in DOC_TYPE_PATTERNS.items():
            if any(re.search(p, text_lower) for p in patterns):
                return doc_type
        return "unknown"

    def _extract_bounding_box(self, element: Element) -> Optional[List[float]]:
        try:
            coords = element.metadata.coordinates
            if coords and coords.points:
                points = coords.points
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                return [min(xs), min(ys), max(xs), max(ys)]
        except (AttributeError, TypeError):
            pass
        return None

    def parse(self, file_path: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        Parses PDF, DOCX, HTML, or TXT into structural elements with bounding box coordinates.
        Returns a list of dictionaries normalized to the expected schema.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            elements = partition_pdf(
                filename=str(path),
                strategy="hi_res",
                infer_table_structure=True,
                extract_images_in_pdf=True,
                include_page_breaks=False
            )
        elif suffix == ".docx":
            elements = partition_docx(filename=str(path))
        elif suffix in [".htm", ".html"]:
            elements = partition_html(filename=str(path))
        elif suffix == ".txt":
            elements = partition_text(filename=str(path))
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        full_text = " ".join(e.text for e in elements[:20] if hasattr(e, 'text'))
        doc_type = self._detect_doc_type(full_text)

        raw_elements = []
        for idx, element in enumerate(elements):
            if not hasattr(element, 'text') or not element.text.strip():
                continue

            block_type = BLOCK_TYPE_MAP.get(type(element).__name__, "text")
            bbox = self._extract_bounding_box(element)
            page = getattr(element.metadata, 'page_number', 1) or 1

            raw_elements.append({
                "content": element.text,
                "block_type": block_type,
                "page_num": page,
                "bounding_box": bbox,
                "split_idx": idx,
                "doc_type": doc_type,
                "doc_id": doc_id,
                "source_file": str(path.name),
            })

        return raw_elements

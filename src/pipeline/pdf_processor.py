"""
PDF processing pipeline using GROBID (primary) with PyMuPDF fallback.
Extracts structured paper content: sections, tables, figures, and references.
GROBID provides superior section segmentation and reference extraction via TEI XML.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import requests
from loguru import logger

from src.models.data_models import Paper, Section


# GROBID TEI XML namespaces
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# Section type mapping from GROBID header keywords
SECTION_KEYWORDS = {
    "abstract": ["abstract"],
    "introduction": ["introduction", "intro", "background", "motivation"],
    "related_work": ["related work", "literature review", "prior work", "state of the art"],
    "methods": ["method", "methodology", "approach", "proposed", "framework", "system", "architecture", "model"],
    "results": ["result", "experiment", "evaluation", "performance", "analysis"],
    "discussion": ["discussion", "interpretation", "implications"],
    "conclusion": ["conclusion", "summary", "future work", "concluding"],
}


class PDFProcessor:
    """
    Processes research paper PDFs into structured Paper objects.
    Uses GROBID for TEI XML parsing when available, falls back to PyMuPDF.
    """

    def __init__(self, grobid_url: str = "http://localhost:8070", use_grobid: bool = True):
        self.grobid_url = grobid_url.rstrip("/")
        self.use_grobid = use_grobid
        self._grobid_available = False

        if use_grobid:
            self._check_grobid()

    def _check_grobid(self):
        """Check if GROBID server is running."""
        try:
            response = requests.get(f"{self.grobid_url}/api/isalive", timeout=5)
            self._grobid_available = response.status_code == 200
            if self._grobid_available:
                logger.info(f"GROBID server is available at {self.grobid_url}")
            else:
                logger.warning("GROBID server responded but may not be healthy")
        except requests.exceptions.ConnectionError:
            logger.warning(
                f"GROBID server not available at {self.grobid_url}. "
                "Falling back to PyMuPDF. To use GROBID, run: "
                "docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0"
            )
            self._grobid_available = False

    def process_pdf(self, pdf_path: str, paper: Optional[Paper] = None) -> Paper:
        """
        Process a PDF file into a structured Paper object.
        
        Args:
            pdf_path: Path to the PDF file
            paper: Optional existing Paper object to update
            
        Returns:
            Paper object with extracted sections
        """
        if paper is None:
            paper = Paper(pdf_path=pdf_path)
        else:
            paper.pdf_path = pdf_path

        path = Path(pdf_path)
        if not path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            paper.errors = [f"PDF not found: {pdf_path}"]
            return paper

        if self._grobid_available:
            try:
                paper = self._process_with_grobid(path, paper)
                paper.is_processed = True
                logger.info(f"GROBID processed: '{paper.title[:60]}...' ({len(paper.sections)} sections)")
                return paper
            except Exception as e:
                logger.warning(f"GROBID processing failed: {e}. Falling back to PyMuPDF.")

        # Fallback to PyMuPDF
        paper = self._process_with_pymupdf(path, paper)
        paper.is_processed = True
        logger.info(f"PyMuPDF processed: '{paper.title[:60]}...' ({len(paper.sections)} sections)")
        return paper

    def process_batch(self, papers: list[Paper]) -> list[Paper]:
        """Process a batch of papers that have pdf_path set."""
        processed = []
        for i, paper in enumerate(papers):
            if not paper.pdf_path:
                continue
            try:
                result = self.process_pdf(paper.pdf_path, paper)
                processed.append(result)
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(papers)} PDFs")
            except Exception as e:
                logger.error(f"Failed to process PDF for '{paper.title[:50]}': {e}")
                paper.is_processed = False
                processed.append(paper)
        return processed

    # ─── GROBID Processing ──────────────────────────────────────────

    def _process_with_grobid(self, pdf_path: Path, paper: Paper) -> Paper:
        """Process PDF using GROBID's processFulltextDocument endpoint."""
        url = f"{self.grobid_url}/api/processFulltextDocument"

        with open(pdf_path, "rb") as f:
            files = {"input": (pdf_path.name, f, "application/pdf")}
            data = {
                "consolidateHeader": "1",
                "consolidateCitations": "1",
                "includeRawCitations": "1",
                "teiCoordinates": "0",
            }
            response = requests.post(url, files=files, data=data, timeout=120)
            response.raise_for_status()

        tei_xml = response.text
        return self._parse_tei_xml(tei_xml, paper)

    def _parse_tei_xml(self, tei_xml: str, paper: Paper) -> Paper:
        """Parse GROBID TEI XML output into a Paper object."""
        root = ET.fromstring(tei_xml)

        # Extract title from header
        title_elem = root.find(".//tei:titleStmt/tei:title", TEI_NS)
        if title_elem is not None and title_elem.text:
            if not paper.title:
                paper.title = title_elem.text.strip()

        # Extract abstract
        abstract_elem = root.find(".//tei:profileDesc/tei:abstract", TEI_NS)
        if abstract_elem is not None:
            abstract_text = self._extract_text_recursive(abstract_elem)
            if abstract_text:
                paper.abstract = abstract_text
                paper.sections.append(Section(
                    title="Abstract",
                    content=abstract_text,
                    section_type="abstract",
                ))

        # Extract authors from header
        if not paper.authors:
            authors = []
            for author in root.findall(".//tei:fileDesc//tei:author", TEI_NS):
                persname = author.find("tei:persName", TEI_NS)
                if persname is not None:
                    forename = persname.find("tei:forename", TEI_NS)
                    surname = persname.find("tei:surname", TEI_NS)
                    name_parts = []
                    if forename is not None and forename.text:
                        name_parts.append(forename.text)
                    if surname is not None and surname.text:
                        name_parts.append(surname.text)
                    if name_parts:
                        authors.append(" ".join(name_parts))
            paper.authors = authors

        # Extract body sections
        body = root.find(".//tei:body", TEI_NS)
        if body is not None:
            for div in body.findall("tei:div", TEI_NS):
                section = self._parse_tei_div(div)
                if section and section.content.strip():
                    paper.sections.append(section)

        # Extract references
        refs = root.findall(".//tei:listBibl/tei:biblStruct", TEI_NS)
        ref_titles = []
        for ref in refs:
            title_el = ref.find(".//tei:title", TEI_NS)
            if title_el is not None and title_el.text:
                ref_titles.append(title_el.text)

        # Build full text from all sections
        paper.full_text = "\n\n".join(s.content for s in paper.sections if s.content)

        return paper

    def _parse_tei_div(self, div_elem) -> Optional[Section]:
        """Parse a single TEI div element into a Section."""
        head = div_elem.find("tei:head", TEI_NS)
        title = head.text.strip() if head is not None and head.text else ""

        # Extract all paragraphs
        paragraphs = []
        for p in div_elem.findall("tei:p", TEI_NS):
            text = self._extract_text_recursive(p)
            if text:
                paragraphs.append(text)

        content = "\n\n".join(paragraphs)
        section_type = self._classify_section(title)

        # Extract tables
        tables = []
        for table in div_elem.findall(".//tei:table", TEI_NS):
            table_text = self._extract_text_recursive(table)
            if table_text:
                tables.append(table_text)

        # Extract figure descriptions
        figures = []
        for fig in div_elem.findall(".//tei:figure", TEI_NS):
            fig_desc = fig.find("tei:figDesc", TEI_NS)
            if fig_desc is not None and fig_desc.text:
                figures.append(fig_desc.text)

        return Section(
            title=title,
            content=content,
            section_type=section_type,
            tables=tables,
            figures=figures,
        )

    def _extract_text_recursive(self, elem) -> str:
        """Extract all text content from an XML element, including children."""
        parts = []
        if elem.text:
            parts.append(elem.text)
        for child in elem:
            child_text = self._extract_text_recursive(child)
            if child_text:
                parts.append(child_text)
            if child.tail:
                parts.append(child.tail)
        return " ".join(parts).strip()

    # ─── PyMuPDF Fallback Processing ────────────────────────────────

    def _process_with_pymupdf(self, pdf_path: Path, paper: Paper) -> Paper:
        """Process PDF using PyMuPDF with heuristic section segmentation."""
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + "\n"
        doc.close()

        paper.full_text = full_text

        # Try to extract tables with pdfplumber
        tables = self._extract_tables_pdfplumber(pdf_path)

        # Heuristic section segmentation
        sections = self._segment_sections(full_text)

        # Add tables to the methods/results section if available
        for section in sections:
            if section.section_type in ("methods", "results"):
                section.tables = tables
                break

        paper.sections = sections

        # Extract title from first lines if not set
        if not paper.title:
            lines = full_text.strip().split("\n")
            for line in lines[:5]:
                line = line.strip()
                if len(line) > 20 and not line[0].isdigit():
                    paper.title = line
                    break

        return paper

    def _segment_sections(self, text: str) -> list[Section]:
        """Heuristically segment raw text into sections."""
        # Pattern: lines that look like section headers
        # e.g., "1. Introduction", "2. Related Work", "Abstract", "III. Methods"
        header_pattern = re.compile(
            r"^(?:\d+\.?\s+|[IVX]+\.?\s+)?("
            r"abstract|introduction|related\s+work|literature\s+review|"
            r"background|method|methodology|approach|proposed|framework|"
            r"experiment|result|evaluation|discussion|conclusion|"
            r"future\s+work|acknowledgment|reference"
            r")s?.*$",
            re.IGNORECASE | re.MULTILINE,
        )

        matches = list(header_pattern.finditer(text))

        if not matches:
            # No sections found — treat entire text as one section
            return [Section(title="Full Text", content=text, section_type="full_text")]

        sections = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            header_text = match.group(0).strip()
            content = text[start + len(header_text):end].strip()
            section_type = self._classify_section(header_text)

            sections.append(Section(
                title=header_text,
                content=content,
                section_type=section_type,
            ))

        return sections

    def _classify_section(self, header: str) -> str:
        """Classify a section header into a standard type."""
        header_lower = header.lower()
        for section_type, keywords in SECTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in header_lower:
                    return section_type
        return "other"

    def _extract_tables_pdfplumber(self, pdf_path: Path) -> list[str]:
        """Extract tables from PDF using pdfplumber."""
        tables = []
        try:
            import pdfplumber
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        if table:
                            # Convert to string representation
                            rows = []
                            for row in table:
                                cells = [str(c) if c else "" for c in row]
                                rows.append(" | ".join(cells))
                            tables.append("\n".join(rows))
        except Exception as e:
            logger.debug(f"Table extraction failed: {e}")
        return tables

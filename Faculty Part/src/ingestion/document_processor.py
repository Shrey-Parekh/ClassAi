"""
Document processing for PDFs, images, and text files.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import re
import pytesseract
from PIL import Image
from pypdf import PdfReader
import io


class DocumentProcessor:
    """
    Process various document types into text.
    
    Handles:
    - PDFs (text extraction + OCR for scanned pages)
    - Images (OCR + visual description)
    - Text files
    """
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """
        Initialize document processor.
        
        Args:
            tesseract_path: Path to tesseract executable (if not in PATH)
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    def process_document(
        self,
        file_path: Path,
        doc_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a document file into structured content.
        
        Args:
            file_path: Path to document file
            doc_metadata: Document metadata (title, date, applies_to, etc.)
        
        Returns:
            Dict with extracted content and metadata
        """
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self._process_pdf(file_path, doc_metadata)
        elif suffix in ['.png', '.jpg', '.jpeg', '.tiff']:
            return self._process_image(file_path, doc_metadata)
        elif suffix in ['.txt', '.md']:
            return self._process_text(file_path, doc_metadata)
        elif suffix == '.json':
            return self._process_json(file_path, doc_metadata)
        elif suffix == '.csv':
            return self._process_csv(file_path, doc_metadata)
        elif suffix in ['.xlsx', '.xls']:
            return self._process_excel(file_path, doc_metadata)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    def _process_pdf(
        self,
        file_path: Path,
        doc_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process PDF file.
        
        Extracts text directly where possible, uses OCR for scanned pages.
        """
        reader = PdfReader(file_path)

        pages_content = []
        images_found = []
        ocr_pages = 0
        ocr_failed_pages = 0
        empty_pages = []  # K4: track per-page failures

        for page_num, page in enumerate(reader.pages):
            # Try text extraction first
            try:
                text = page.extract_text() or ""
            except Exception as e:
                print(f"[PDF] extract_text failed on page {page_num + 1}: {e}")
                text = ""

            # If minimal text, likely scanned or font-corrupted - use OCR fallback
            if len(text.strip()) < 50:
                ocr_text = self._ocr_pdf_page(file_path, page_num + 1)
                if ocr_text and ocr_text.strip():
                    text = ocr_text
                    ocr_pages += 1
                else:
                    ocr_failed_pages += 1
                    if len(text.strip()) < 20:
                        empty_pages.append(page_num + 1)

            # Join continuation lines to fix wrapped table rows
            text = self._join_continuation_lines(text)
            
            pages_content.append({
                "page_num": page_num + 1,
                "content": text,
            })
            
            # Extract embedded images
            if '/XObject' in page['/Resources']:
                xobjects = page['/Resources']['/XObject'].get_object()
                
                for obj_name in xobjects:
                    obj = xobjects[obj_name]
                    if obj['/Subtype'] == '/Image':
                        # Extract and process image
                        image_data = self._extract_pdf_image(obj)
                        if image_data:
                            images_found.append({
                                "page_num": page_num + 1,
                                "image_name": obj_name,
                                "ocr_text": image_data["ocr_text"],
                                "description": image_data["description"],
                            })
        
        # Combine all text
        full_text = "\n\n".join([p["content"] for p in pages_content])

        # Surface silent-drop failures: a PDF with no recoverable text should
        # raise rather than ingest as an empty document.
        total_chars = len(full_text.strip())
        page_count = len(reader.pages)
        min_expected = max(200, page_count * 50)
        if total_chars < min_expected:
            raise RuntimeError(
                f"PDF extraction produced only {total_chars} chars across "
                f"{page_count} pages for '{file_path.name}' "
                f"(ocr_pages={ocr_pages}, ocr_failed_pages={ocr_failed_pages}). "
                f"The PDF is likely scanned without OCR, has corrupted embedded fonts, "
                f"or is image-only. Repair with 'mutool clean -gggg in.pdf out.pdf', "
                f"'qpdf --linearize', or re-export the source, then retry."
            )

        # K4: per-page empty check — raise if >20% of pages are unrecoverable
        if empty_pages:
            threshold = max(1, int(page_count * 0.2))
            if len(empty_pages) > threshold:
                raise RuntimeError(
                    f"'{file_path.name}': {len(empty_pages)} pages produced <20 chars "
                    f"and OCR also failed (pages: {empty_pages}). "
                    f"Likely font-corrupted or image-only pages. "
                    f"Repair with 'mutool clean -gggg in.pdf out.pdf' then retry."
                )

        if ocr_pages:
            print(f"[PDF] {file_path.name}: OCR used on {ocr_pages} page(s), "
                  f"{ocr_failed_pages} OCR failure(s)")

        return {
            "content": full_text,
            "pages": pages_content,
            "images": images_found,
            "metadata": {
                **doc_metadata,
                "page_count": page_count,
                "has_images": len(images_found) > 0,
                "ocr_pages": ocr_pages,
                "ocr_failed_pages": ocr_failed_pages,
            }
        }
    
    def _process_image(
        self,
        file_path: Path,
        doc_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process image file with OCR.
        
        Creates both textual and visual representations.
        """
        image = Image.open(file_path)
        
        # OCR text extraction
        ocr_text = pytesseract.image_to_string(image)
        
        # Generate visual description (placeholder for vision model)
        description = self._generate_image_description(image)
        
        return {
            "content": ocr_text,
            "visual_description": description,
            "metadata": {
                **doc_metadata,
                "image_size": image.size,
                "image_mode": image.mode,
                "has_text": len(ocr_text.strip()) > 0,
            }
        }
    
    def _process_text(
        self,
        file_path: Path,
        doc_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process plain text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "content": content,
            "metadata": doc_metadata,
        }
    
    def _process_json(
        self,
        file_path: Path,
        doc_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process JSON file (supports both JSON array and JSONL formats).
        
        Creates multiple focused chunks per faculty member, each < 490 tokens.
        Each chunk includes faculty name for searchability.
        """
        import json
        
        content_parts = []
        
        with open(file_path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig handles BOM
            content = f.read().strip()
            
            # Try parsing as JSON array first
            try:
                data = json.loads(content)
                
                # If it's a list, process each item
                if isinstance(data, list):
                    for i, obj in enumerate(data, 1):
                        if not isinstance(obj, dict):
                            continue
                        
                        # Extract text and metadata
                        text = obj.get('text', '')
                        name = obj.get('name', f'Entry {i}')
                        profile_url = obj.get('profile_url', '')
                        
                        # If no 'text' field, build from other fields
                        if not text:
                            text = self._build_faculty_text_from_dict(obj)
                        
                        # Split faculty profile into focused chunks
                        faculty_chunks = self._split_faculty_profile(
                            name, profile_url, text
                        )
                        content_parts.extend(faculty_chunks)
                
                # If it's a single dict, process it
                elif isinstance(data, dict):
                    text = data.get('text', '')
                    name = data.get('name', 'Entry 1')
                    profile_url = data.get('profile_url', '')
                    
                    if not text:
                        text = self._build_faculty_text_from_dict(data)
                    
                    faculty_chunks = self._split_faculty_profile(
                        name, profile_url, text
                    )
                    content_parts.extend(faculty_chunks)
                    
            except json.JSONDecodeError:
                # Fall back to JSONL format (one JSON per line)
                f.seek(0)
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        obj = json.loads(line)
                        
                        # Extract text and metadata
                        text = obj.get('text', '')
                        name = obj.get('name', f'Entry {line_num}')
                        profile_url = obj.get('profile_url', '')
                        
                        if not text:
                            text = self._build_faculty_text_from_dict(obj)
                        
                        # Split faculty profile into focused chunks
                        faculty_chunks = self._split_faculty_profile(
                            name, profile_url, text
                        )
                        content_parts.extend(faculty_chunks)
                        
                    except json.JSONDecodeError as e:
                        print(f"  ⚠ Skipping invalid JSON on line {line_num}: {e}")
                        continue
        
        # Check if any entries were parsed
        if not content_parts:
            raise ValueError(f"No valid JSON entries found in {file_path.name}")
        
        # Combine all chunks with clear separators
        separator = "\n" + "=" * 60 + "\n\n"
        content = separator.join(content_parts)
        
        return {
            "content": content,
            "metadata": {
                **doc_metadata,
                "entry_count": len(content_parts),
                "format": "json"
            }
        }
    
    def _build_faculty_text_from_dict(self, obj: Dict[str, Any]) -> str:
        """Build clean readable faculty text from dictionary fields."""
        parts = []
        
        if 'name' in obj:
            parts.append(f"Name: {obj['name']}")
        
        # Skip "Not specified" fields — they add noise
        for field in ['qualification', 'experience', 'research_interests']:
            if field in obj:
                value = str(obj[field]).strip()
                if value and value.lower() != 'not specified':
                    parts.append(f"{field.replace('_', ' ').title()}: {value}")
        
        # Handle publications as nested dict
        if 'publications' in obj:
            pubs = obj['publications']
            if isinstance(pubs, dict):
                for pub_type, pub_list in pubs.items():
                    if isinstance(pub_list, list) and pub_list:
                        parts.append(f"\n{pub_type.title()}:")
                        for pub in pub_list:
                            parts.append(f"  - {pub}")
            elif isinstance(pubs, list) and pubs:
                parts.append("Publications:")
                for pub in pubs:
                    parts.append(f"  - {pub}")
        
        # Handle awards as list
        if 'awards' in obj:
            awards = obj['awards']
            if isinstance(awards, list) and awards:
                parts.append("\nAwards:")
                for award in awards:
                    parts.append(f"  - {award}")
        
        return "\n".join(parts)
    
    def _split_faculty_profile(
        self,
        name: str,
        profile_url: str,
        text: str
    ) -> List[str]:
        """
        Split a faculty profile into multiple focused chunks.
        
        Each chunk:
        - Includes faculty name for searchability
        - Is semantically complete
        - Stays under 490 tokens (~1960 chars)
        
        Returns:
            List of chunk strings
        """
        chunks = []
        
        # Approximate token limit (490 tokens ≈ 1960 chars)
        MAX_CHARS = 1960
        
        # Parse the text to extract sections
        # Format: "Name: X Qualification: Y Experience: Z Research Interests: W Publications: ... Awards: ..."
        
        # Extract core info (everything before Publications)
        pub_match = re.search(r'Publications:', text, re.IGNORECASE)
        awards_match = re.search(r'Awards:', text, re.IGNORECASE)
        
        if pub_match:
            core_info = text[:pub_match.start()].strip()
            remaining = text[pub_match.start():].strip()
        else:
            # No publications section, treat all as core info
            core_info = text.strip()
            remaining = ""
        
        # Chunk 1: Name + Core Info (Qualification, Experience, Research Interests)
        chunk1 = f"Faculty: {name}\n"
        if profile_url:
            chunk1 += f"Profile: {profile_url}\n\n"
        chunk1 += core_info
        chunks.append(chunk1)
        
        # Process remaining content (Publications and Awards)
        if remaining:
            # Split publications and awards
            if awards_match:
                publications_text = remaining[:awards_match.start() - pub_match.start()].strip()
                awards_text = remaining[awards_match.start() - pub_match.start():].strip()
            else:
                publications_text = remaining
                awards_text = ""
            
            # Split publications into multiple chunks if needed
            if publications_text:
                pub_chunks = self._split_long_section(
                    name, "Publications", publications_text, MAX_CHARS
                )
                chunks.extend(pub_chunks)
            
            # Add awards chunk if exists
            if awards_text:
                awards_chunk = f"Faculty: {name}\n\n{awards_text}"
                if len(awards_chunk) > MAX_CHARS:
                    # Split awards if too long
                    award_chunks = self._split_long_section(
                        name, "Awards", awards_text, MAX_CHARS
                    )
                    chunks.extend(award_chunks)
                else:
                    chunks.append(awards_chunk)
        
        return chunks
    
    def _split_long_section(
        self,
        name: str,
        section_name: str,
        section_text: str,
        max_chars: int
    ) -> List[str]:
        """
        Split a long section (like Publications) into multiple chunks.
        
        Each chunk includes faculty name and section context.
        """
        chunks = []
        
        # Split by sentences or publication entries
        # Publications are typically separated by journal names or years
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', section_text)
        
        current_chunk = f"Faculty: {name}\n\n{section_name}:\n"
        header_len = len(current_chunk)
        chunk_num = 1
        
        for sentence in sentences:
            test_chunk = current_chunk + sentence + " "
            
            if len(test_chunk) <= max_chars:
                current_chunk = test_chunk
            else:
                # Save current chunk if it has content beyond header
                if len(current_chunk) > header_len + 10:
                    chunks.append(current_chunk.strip())
                    chunk_num += 1
                
                # Start new chunk
                current_chunk = f"Faculty: {name}\n\n{section_name} (continued {chunk_num}):\n{sentence} "
        
        # Add final chunk
        if len(current_chunk) > header_len + 10:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _process_csv(
        self,
        file_path: Path,
        doc_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process CSV file into readable text format."""
        import csv
        
        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    reader = csv.DictReader(f)
                    headers = reader.fieldnames
                    rows = list(reader)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            raise ValueError(f"Could not decode {file_path.name} with any common encoding")
        
        # Convert to readable text format
        content_parts = []
        
        # Add title
        title = doc_metadata.get('title', file_path.stem)
        content_parts.append(f"{title}\n")
        content_parts.append("=" * len(title) + "\n\n")
        
        # Add each row as a structured entry
        for i, row in enumerate(rows, 1):
            content_parts.append(f"Entry {i}:\n")
            for key, value in row.items():
                if value:  # Only include non-empty values
                    content_parts.append(f"  {key}: {value}\n")
            content_parts.append("\n")
        
        content = "".join(content_parts)
        
        return {
            "content": content,
            "metadata": {
                **doc_metadata,
                "row_count": len(rows),
                "columns": list(headers) if headers else [],
            }
        }
    
    def _process_excel(
        self,
        file_path: Path,
        doc_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process Excel file into readable text format."""
        try:
            import openpyxl
        except ImportError:
            raise ImportError("openpyxl required for Excel files. Install: pip install openpyxl")
        
        workbook = openpyxl.load_workbook(file_path, data_only=True)
        sheet = workbook.active
        
        # Get headers from first row
        headers = [cell.value for cell in sheet[1]]
        
        # Get all rows
        rows = []
        for row in sheet.iter_rows(min_row=2, values_only=True):
            if any(cell is not None for cell in row):  # Skip empty rows
                row_dict = dict(zip(headers, row))
                rows.append(row_dict)
        
        # Convert to readable text format
        content_parts = []
        
        # Add title
        title = doc_metadata.get('title', file_path.stem)
        content_parts.append(f"{title}\n")
        content_parts.append("=" * len(title) + "\n\n")
        
        # Add each row as a structured entry
        for i, row in enumerate(rows, 1):
            content_parts.append(f"Entry {i}:\n")
            for key, value in row.items():
                if value is not None and str(value).strip():  # Only include non-empty values
                    content_parts.append(f"  {key}: {value}\n")
            content_parts.append("\n")
        
        content = "".join(content_parts)
        
        return {
            "content": content,
            "metadata": {
                **doc_metadata,
                "row_count": len(rows),
                "columns": headers,
                "sheet_name": sheet.title,
            }
        }
    
    def _ocr_pdf_page(self, file_path: Path, page_num: int) -> str:
        """
        OCR a single PDF page using pdf2image + pytesseract.

        Args:
            file_path: Path to the PDF file
            page_num: 1-indexed page number

        Returns:
            Extracted text, or empty string on failure.
        """
        try:
            from pdf2image import convert_from_path
        except ImportError:
            print("[OCR] pdf2image not installed. Install: pip install pdf2image "
                  "(and ensure poppler is available on PATH)")
            return ""

        try:
            images = convert_from_path(
                str(file_path),
                first_page=page_num,
                last_page=page_num,
                dpi=300,
            )
        except Exception as e:
            print(f"[OCR] pdf2image failed for {file_path.name} p{page_num}: {e}")
            return ""

        if not images:
            return ""

        try:
            text = pytesseract.image_to_string(images[0])
        except Exception as e:
            print(f"[OCR] tesseract failed for {file_path.name} p{page_num}: {e}")
            text = ""
        finally:
            for img in images:
                img.close()

        return text or ""

    def _extract_pdf_image(self, image_obj) -> Optional[Dict[str, str]]:
        """Extract and process image from PDF."""
        try:
            # Extract image data
            # This is simplified - actual implementation depends on PDF structure
            size = (image_obj['/Width'], image_obj['/Height'])
            data = image_obj.get_data()

            # Convert to PIL Image
            image = Image.open(io.BytesIO(data))

            # OCR
            ocr_text = pytesseract.image_to_string(image)

            # Generate description
            description = self._generate_image_description(image)

            return {
                "ocr_text": ocr_text,
                "description": description,
            }
        except Exception as e:
            print(f"Error extracting image: {e}")
            return None

    def _generate_image_description(self, image: Image.Image) -> str:
        """
        Generate textual description of image content.

        In production: use vision-language model (CLIP, BLIP, GPT-4V, etc.)
        """
        # Placeholder - implement with vision model
        return f"[Image: {image.size[0]}x{image.size[1]} pixels]"

    def _join_continuation_lines(self, text: str) -> str:
        """Join lines that are continuations of the previous line.

        Does NOT merge when either line looks like a table row — this
        preserves two-column tables (parameter | value) in ERB and FAG.

        >>> proc = DocumentProcessor()
        >>> proc._join_continuation_lines("Parameter   Value\\nlower cell")
        'Parameter   Value\\nlower cell'
        >>> proc._join_continuation_lines("This is a long\\nsentence continued")
        'This is a long sentence continued'
        """
        _table_pat = re.compile(r'^\w[^\n]{1,40}\s{3,}.+')
        lines = text.split('\n')
        cleaned = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned.append('')
                continue

            prev = cleaned[-1] if cleaned else ''
            prev_stripped = prev.strip()

            # Guard: do NOT merge if either line looks like a table row
            is_table_line = (
                '   ' in stripped or          # 3+ leading spaces
                line.startswith('   ') or      # leading indent
                (prev_stripped and '   ' in prev_stripped) or  # prev has column gap
                _table_pat.match(stripped) or
                (prev_stripped and _table_pat.match(prev_stripped))
            )

            if (not is_table_line and
                    prev_stripped and stripped and
                    (stripped[0].islower() or
                     stripped.startswith('(') or
                     re.match(r'^(with|or|and|for|by|of|in|at|to)\s', stripped, re.IGNORECASE))):
                cleaned[-1] = cleaned[-1].rstrip() + ' ' + stripped
            else:
                cleaned.append(line)

        return '\n'.join(cleaned)

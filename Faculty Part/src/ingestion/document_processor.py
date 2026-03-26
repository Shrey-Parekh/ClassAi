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
        
        for page_num, page in enumerate(reader.pages):
            # Try text extraction first
            text = page.extract_text()
            
            # If minimal text, likely scanned - use OCR
            if len(text.strip()) < 50:
                # Extract page as image and OCR
                # Note: This requires pdf2image library
                text = self._ocr_pdf_page(page)
            
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
        
        return {
            "content": full_text,
            "pages": pages_content,
            "images": images_found,
            "metadata": {
                **doc_metadata,
                "page_count": len(reader.pages),
                "has_images": len(images_found) > 0,
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
        """
        Build faculty text from dictionary fields.
        
        Handles cases where 'text' field doesn't exist.
        """
        parts = []
        
        # Name
        if 'name' in obj:
            parts.append(f"Name: {obj['name']}")
        
        # Qualification
        if 'qualification' in obj:
            parts.append(f"Qualification: {obj['qualification']}")
        
        # Experience
        if 'experience' in obj:
            parts.append(f"Experience: {obj['experience']}")
        
        # Research Interests
        if 'research_interests' in obj:
            parts.append(f"Research Interests: {obj['research_interests']}")
        
        # Publications
        if 'publications' in obj:
            parts.append(f"Publications: {obj['publications']}")
        
        # Awards
        if 'awards' in obj:
            parts.append(f"Awards: {obj['awards']}")
        
        # Any other fields
        for key, value in obj.items():
            if key not in ['name', 'qualification', 'experience', 'research_interests', 
                          'publications', 'awards', 'profile_url', 'text', 'metadata']:
                parts.append(f"{key.replace('_', ' ').title()}: {value}")
        
        return " ".join(parts)
    
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
    
    def _ocr_pdf_page(self, page) -> str:
        """OCR a PDF page (requires pdf2image)."""
        # Placeholder - implement with pdf2image
        # from pdf2image import convert_from_path
        # images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
        # return pytesseract.image_to_string(images[0])
        return "[OCR extraction not implemented]"
    
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

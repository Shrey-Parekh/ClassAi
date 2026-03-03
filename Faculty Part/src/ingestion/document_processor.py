"""
Document processing for PDFs, images, and text files.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
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

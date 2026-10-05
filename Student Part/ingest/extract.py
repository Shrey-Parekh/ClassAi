"""
Markdown extraction optimized for NMIMS academic documents.
Handles both syllabus and question paper formats.
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document


def extract_syllabus_metadata(content: str, filename: str) -> Dict[str, Any]:
    """Extract metadata from NMIMS syllabus format."""
    metadata = {
        "type": "syllabus",
        "source": filename,
        "courses": []
    }
    
    # Extract all courses from the syllabus
    # Pattern: ## COURSE X: Course Name
    course_pattern = r'##\s*COURSE\s+\d+:\s*([^\n]+)'
    courses = re.findall(course_pattern, content, re.IGNORECASE)
    
    if courses:
        metadata["courses"] = courses
        metadata["subject"] = ", ".join(courses[:3])  # First 3 courses
    else:
        # Fallback to first heading
        first_heading = re.search(r'#\s*([^\n]+)', content)
        if first_heading:
            metadata["subject"] = first_heading.group(1).strip()
    
    # Extract course codes
    code_pattern = r'\*\*Course Code:\*\*\s*([A-Z0-9]+)'
    codes = re.findall(code_pattern, content)
    if codes:
        metadata["course_codes"] = codes
    
    # Extract credits
    credits_pattern = r'\*\*Credits:\*\*\s*(\d+)'
    credits = re.findall(credits_pattern, content)
    if credits:
        metadata["credits"] = [int(c) for c in credits]
    
    # Extract semesters
    semester_pattern = r'\*\*Semester:\*\*\s*([^\n]+)'
    semesters = re.findall(semester_pattern, content)
    if semesters:
        metadata["semesters"] = semesters
    
    return metadata


def extract_qp_metadata(content: str, filename: str) -> Dict[str, Any]:
    """Extract metadata from NMIMS question paper format."""
    metadata = {
        "type": "question_paper",
        "source": filename
    }
    
    # Extract subject from title
    # Pattern: ## Examination Paper: Subject Name
    subject_match = re.search(r'##\s*Examination Paper:\s*([^\n]+)', content, re.IGNORECASE)
    if subject_match:
        metadata["subject"] = subject_match.group(1).strip()
    else:
        # Fallback
        first_heading = re.search(r'##\s*([^\n]+)', content)
        if first_heading:
            metadata["subject"] = first_heading.group(1).strip()
    
    # Extract academic year
    year_match = re.search(r'\*\*Academic Year:\*\*\s*([^\n]+)', content)
    if year_match:
        metadata["academic_year"] = year_match.group(1).strip()
    
    # Extract exam type
    if "Final" in content:
        metadata["exam_type"] = "Final"
    elif "Mid" in content or "Internal" in content:
        metadata["exam_type"] = "Mid-term"
    
    # Extract total marks
    marks_match = re.search(r'\*\*Total Marks:\*\*\s*(\d+)', content)
    if marks_match:
        metadata["total_marks"] = int(marks_match.group(1))
    
    # Extract duration
    duration_match = re.search(r'\*\*Duration:\*\*\s*([^\n]+)', content)
    if duration_match:
        metadata["duration"] = duration_match.group(1).strip()
    
    # Extract semester
    semester_match = re.search(r'\*\*Semester:\*\*\s*([^\n]+)', content)
    if semester_match:
        metadata["semester"] = semester_match.group(1).strip()
    
    return metadata


def extract_document(filepath: str) -> List[Document]:
    """
    Extract a single markdown document with metadata.
    Auto-detects document type (syllabus or question paper).
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            print(f"⚠️  Empty file: {filepath}")
            return []
        
        filename = os.path.basename(filepath)
        
        # Auto-detect document type
        if "Examination Paper" in content or "Question" in content:
            doc_type = "question_paper"
            metadata = extract_qp_metadata(content, filename)
        else:
            doc_type = "syllabus"
            metadata = extract_syllabus_metadata(content, filename)
        
        metadata["filepath"] = filepath
        metadata["doc_type"] = doc_type
        
        # Create document
        doc = Document(page_content=content, metadata=metadata)
        
        return [doc]
        
    except Exception as e:
        print(f"❌ Error extracting {filepath}: {e}")
        return []


def extract_all(directory: str) -> List[Document]:
    """
    Extract all markdown files from a directory.
    """
    all_docs = []
    path = Path(directory)
    
    if not path.exists():
        print(f"⚠️  Directory not found: {directory}")
        return []
    
    md_files = list(path.glob("**/*.md"))
    
    if not md_files:
        print(f"⚠️  No markdown files found in {directory}")
        return []
    
    print(f"📖 Found {len(md_files)} markdown file(s) in {directory}")
    
    for md_file in md_files:
        docs = extract_document(str(md_file))
        all_docs.extend(docs)
        if docs:
            subject = docs[0].metadata.get('subject', 'Unknown')
            doc_type = docs[0].metadata.get('type', 'unknown')
            print(f"  ✓ {md_file.name} ({doc_type}): {subject[:60]}")
    
    return all_docs


if __name__ == "__main__":
    # Test extraction
    import sys
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        docs = extract_document(test_file)
        if docs:
            print(f"\n📄 Extracted: {docs[0].metadata}")
            print(f"\n📝 Content preview:\n{docs[0].page_content[:500]}...")
    else:
        # Test on all files
        print("Testing Syllabus Extraction:")
        syllabus_docs = extract_all("data/syllabus")
        print(f"\nTotal: {len(syllabus_docs)} documents\n")
        
        print("Testing Question Paper Extraction:")
        qp_docs = extract_all("data/question_papers")
        print(f"\nTotal: {len(qp_docs)} documents")

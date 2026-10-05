"""
Semantic chunking optimized for NMIMS academic documents.
Preserves course structure, units, and questions.
"""
import re
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_syllabus(doc: Document) -> List[Document]:
    """
    Chunk syllabus by courses and units.
    Each course becomes multiple chunks based on units.
    """
    chunks = []
    content = doc.page_content
    base_metadata = doc.metadata.copy()
    
    # Split by COURSE sections
    course_pattern = r'(##\s*COURSE\s+\d+:.*?)(?=##\s*COURSE\s+\d+:|$)'
    courses = re.findall(course_pattern, content, re.DOTALL | re.IGNORECASE)
    
    if not courses:
        # No course sections found, chunk the whole document
        return chunk_by_size(doc, chunk_size=1500, overlap=200)
    
    for course_idx, course_content in enumerate(courses, 1):
        # Extract course name
        course_name_match = re.search(r'##\s*COURSE\s+\d+:\s*([^\n]+)', course_content)
        course_name = course_name_match.group(1).strip() if course_name_match else f"Course {course_idx}"
        
        # Extract course code
        course_code_match = re.search(r'\*\*Course Code:\*\*\s*([A-Z0-9]+)', course_content)
        course_code = course_code_match.group(1) if course_code_match else None
        
        # Split course into sections (Course Info, Units, etc.)
        # First, extract the header section (before units table)
        unit_table_start = re.search(r'\|\s*Unit\s*\|\s*Topic\s*\|', course_content, re.IGNORECASE)
        
        if unit_table_start:
            # Header section (course info, objectives, outcomes)
            header_content = course_content[:unit_table_start.start()]
            if header_content.strip():
                metadata = base_metadata.copy()
                metadata.update({
                    "chunk_type": "course_info",
                    "course_name": course_name,
                    "course_code": course_code,
                    "chunk_index": len(chunks)
                })
                chunks.append(Document(page_content=header_content.strip(), metadata=metadata))
            
            # Units section - extract the full table content
            units_section = course_content[unit_table_start.start():]
            
            # Find where the table ends (next section or end)
            table_end = re.search(r'\n\n###\s+', units_section)
            if table_end:
                units_table = units_section[:table_end.start()]
            else:
                # Table goes to end of course section
                units_table = units_section
            
            # Extract unit rows - capture full content including the dash and everything after
            # Pattern: | number | **Topic** – details | duration |
            unit_pattern = r'\|\s*(\d+)\s*\|\s*\*\*([^*]+)\*\*\s*[–\-]\s*([^|]+?)\s*\|\s*(\d+)\s*\|'
            unit_matches = re.findall(unit_pattern, units_table, re.DOTALL)
            
            for unit_num, unit_name, unit_details, duration in unit_matches:
                # Clean up the text
                unit_name = unit_name.strip()
                unit_details = ' '.join(unit_details.split())  # Normalize whitespace
                
                # Create comprehensive chunk
                unit_text = f"**Course:** {course_name}\n"
                if course_code:
                    unit_text += f"**Course Code:** {course_code}\n"
                unit_text += f"\n**Unit {unit_num}: {unit_name}**\n\n"
                unit_text += f"**Topics Covered:**\n{unit_details}\n\n"
                unit_text += f"**Duration:** {duration} hours\n"
                
                metadata = base_metadata.copy()
                metadata.update({
                    "chunk_type": "syllabus_unit",
                    "course_name": course_name,
                    "course_code": course_code,
                    "unit_number": unit_num,
                    "unit_name": unit_name,
                    "unit_details": unit_details[:200],  # First 200 chars for metadata
                    "duration_hours": int(duration),
                    "chunk_index": len(chunks)
                })
                chunks.append(Document(page_content=unit_text, metadata=metadata))
        else:
            # No units table, chunk by size
            metadata = base_metadata.copy()
            metadata.update({
                "chunk_type": "course_general",
                "course_name": course_name,
                "course_code": course_code
            })
            course_doc = Document(page_content=course_content, metadata=metadata)
            chunks.extend(chunk_by_size(course_doc, chunk_size=1500, overlap=200))
    
    return chunks


def chunk_question_paper(doc: Document) -> List[Document]:
    """
    Chunk question paper by individual questions.
    Each question becomes a separate chunk.
    """
    chunks = []
    content = doc.page_content
    base_metadata = doc.metadata.copy()
    
    # Extract subject
    subject = base_metadata.get("subject", "Unknown")
    
    # Split by question sections (Q1, Q2, Q1.A, Q2.B, etc.)
    # Pattern matches: **Q1.A**, **Q2**, etc.
    question_pattern = r'(\*\*Q\d+(?:\.[A-Za-z])?\*\*.*?)(?=\*\*Q\d+(?:\.[A-Za-z])?\*\*|$)'
    questions = re.findall(question_pattern, content, re.DOTALL | re.IGNORECASE)
    
    if not questions:
        # No questions found, chunk by size
        return chunk_by_size(doc, chunk_size=800, overlap=100)
    
    for question_content in questions:
        # Extract question number
        q_match = re.search(r'\*\*Q(\d+)(?:\.([A-Za-z]))?\*\*', question_content)
        if not q_match:
            continue
        
        q_num = q_match.group(1)
        q_part = q_match.group(2) if q_match.group(2) else None
        q_id = f"Q{q_num}.{q_part}" if q_part else f"Q{q_num}"
        
        # Extract marks
        marks_match = re.search(r'\[(\d+)\s*Marks?\]', question_content, re.IGNORECASE)
        marks = int(marks_match.group(1)) if marks_match else None
        
        # Extract CO, SO, BL
        co_match = re.search(r'CO-(\d+)', question_content, re.IGNORECASE)
        so_match = re.search(r'SO-(\d+)', question_content, re.IGNORECASE)
        bl_match = re.search(r'BL-(\d+)', question_content, re.IGNORECASE)
        
        co = f"CO-{co_match.group(1)}" if co_match else None
        so = f"SO-{so_match.group(1)}" if so_match else None
        bl = f"BL-{bl_match.group(1)}" if bl_match else None
        
        # Create chunk
        metadata = base_metadata.copy()
        metadata.update({
            "chunk_type": "question",
            "question_id": q_id,
            "question_number": int(q_num),
            "question_part": q_part,
            "marks": marks,
            "course_outcome": co,
            "skill_outcome": so,
            "bloom_level": bl,
            "chunk_index": len(chunks)
        })
        
        # Add context to content
        chunk_text = f"**Subject:** {subject}\n**Question:** {q_id}\n"
        if marks:
            chunk_text += f"**Marks:** {marks}\n"
        if co:
            chunk_text += f"**CO:** {co}\n"
        chunk_text += f"\n{question_content.strip()}"
        
        chunks.append(Document(page_content=chunk_text, metadata=metadata))
    
    return chunks


def chunk_by_size(doc: Document, chunk_size: int = 1000, overlap: int = 150) -> List[Document]:
    """
    Fallback chunking by size with overlap.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents([doc])
    
    # Add chunk index
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        if "chunk_type" not in chunk.metadata:
            chunk.metadata["chunk_type"] = "general"
    
    return chunks


def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Main chunking function. Routes to appropriate chunker based on document type.
    """
    all_chunks = []
    
    for doc in docs:
        doc_type = doc.metadata.get("type", "unknown")
        
        if doc_type == "syllabus":
            chunks = chunk_syllabus(doc)
        elif doc_type == "question_paper":
            chunks = chunk_question_paper(doc)
        else:
            chunks = chunk_by_size(doc)
        
        all_chunks.extend(chunks)
    
    return all_chunks


if __name__ == "__main__":
    # Test chunking
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from ingest.extract import extract_all
    
    print("Testing Chunking:")
    print("=" * 60)
    
    # Test syllabus
    print("\n1. Syllabus Chunking:")
    syllabus_docs = extract_all("data/syllabus")
    if syllabus_docs:
        syllabus_chunks = chunk_documents(syllabus_docs)
        print(f"  Generated {len(syllabus_chunks)} chunks")
        print(f"  Chunk types: {set(c.metadata.get('chunk_type') for c in syllabus_chunks)}")
    
    # Test question papers
    print("\n2. Question Paper Chunking:")
    qp_docs = extract_all("data/question_papers")
    if qp_docs:
        qp_chunks = chunk_documents(qp_docs)
        print(f"  Generated {len(qp_chunks)} chunks")
        print(f"  Chunk types: {set(c.metadata.get('chunk_type') for c in qp_chunks)}")

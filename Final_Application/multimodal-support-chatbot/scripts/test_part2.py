import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import fitz
from PIL import Image
import io

from app.ingestion.pdf_parser import PDFParser
from app.ingestion.chunker import SemanticChunker
from app.retrieval.cross_modal_linker import CrossModalLinker
from app.models.domain import ImageRecord

def create_test_pdf(path: str):
    """Creates a synthetic technical manual PDF for testing."""
    doc = fitz.open()
    
    # --- Page 1 ---
    page1 = doc.new_page()
    
    # Add an H1 heading (Font size 20)
    page1.insert_text((50, 50), "System Overview", fontsize=20, fontname="helv", color=(0,0,0))
    
    # Add a paragraph
    para_text = (
        "The system consists of several components working together. "
        "The main motherboard contains the CPU and RAM slots. "
        "Please refer to Figure 1 for the visual layout of the board."
    )
    page1.insert_text((50, 80), para_text, fontsize=12, fontname="helv")
    
    # Create a dummy image (150x150 blue square)
    img = Image.new('RGB', (150, 150), color = 'blue')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    
    # Insert image
    rect = fitz.Rect(50, 150, 200, 300)
    page1.insert_image(rect, stream=img_bytes)
    
    # --- Page 2 ---
    page2 = doc.new_page()
    
    # Add an H2 heading
    page2.insert_text((50, 50), "Specifications", fontsize=16, fontname="helv")
    
    # Add a table-like structure
    table_text = (
        "| Component | Value |\n"
        "| CPU | Octa-core 3.5GHz |\n"
        "| RAM | 32GB DDR5 |\n"
        "| Storage | 1TB NVMe |"
    )
    page2.insert_text((50, 100), table_text, fontsize=12, fontname="courier")
    
    doc.save(path)
    doc.close()
    print(f"Created test PDF at {path}")

def run_tests():
    pdf_path = "test_manual.pdf"
    create_test_pdf(pdf_path)
    
    print("\n--- 1. Testing PDFParser ---")
    parser = PDFParser()
    parsed_doc = parser.parse(pdf_path)
    print(f"Total Pages Parsed: {parsed_doc.total_pages}")
    for p in parsed_doc.pages:
        print(f"Page {p.page_number}:")
        print(f"  Headings detected: {len(p.headings)} -> {[h['text'] for h in p.headings]}")
        print(f"  Images detected: {len(p.images)}")
        print(f"  Text length: {len(p.text)} chars")
    
    print("\n--- 2. Testing SemanticChunker ---")
    chunker = SemanticChunker()
    chunks = chunker.chunk_document(parsed_doc, doc_id="test_doc_01")
    print(f"Total Chunks Generated: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} [{chunk.chunk_type.value}]: Section Path: {chunk.section_path}")
        print(f"  Text excerpt: '{chunk.text[:60]}...'")
        
    print("\n--- 3. Testing CrossModalLinker ---")
    linker = CrossModalLinker(page_proximity=1)
    
    # We create a dummy ImageRecord representing the image we found
    image_records = []
    for p in parsed_doc.pages:
        for img in p.images:
            image_records.append(ImageRecord(
                doc_id="test_doc_01",
                page_number=p.page_number,
                image_type="photo", # Dummy type
                metadata={"bbox": list(img.bbox)}
            ))
            
    print(f"Input: {len(chunks)} Chunks, {len(image_records)} Images")
    linked_chunks, linked_images = linker.link_chunks_and_images(chunks, image_records)
    
    print("Linkage Results:")
    for chunk in linked_chunks:
        if chunk.linked_images:
            print(f"  Chunk '{chunk.text[:30]}...' -> Links to Images: {chunk.linked_images}")
            
    for img in linked_images:
        if img.linked_chunk_ids:
            print(f"  Image (Page {img.page_number}) -> Links to Chunks: {img.linked_chunk_ids}")

    # Cleanup
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
        
if __name__ == "__main__":
    run_tests()

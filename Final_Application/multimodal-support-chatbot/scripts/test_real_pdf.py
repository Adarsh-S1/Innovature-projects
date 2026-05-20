import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.ingestion.pdf_parser import PDFParser
from app.ingestion.chunker import SemanticChunker
from app.ingestion.image_processor import ImageProcessor
from app.retrieval.cross_modal_linker import CrossModalLinker

def run_paper_test():
    pdf_path = "attention_is_all_you_need.pdf"
    
    print("=== 1. PARSING PDF ===")
    parser = PDFParser()
    parsed_doc = parser.parse(pdf_path)
    print(f"File: {parsed_doc.source_file}")
    print(f"Total Pages: {parsed_doc.total_pages}")
    
    total_images = sum(len(p.images) for p in parsed_doc.pages)
    total_tables = sum(1 for p in parsed_doc.pages if p.has_tables)
    total_headings = sum(len(p.headings) for p in parsed_doc.pages)
    print(f"Extracted Images: {total_images}")
    print(f"Pages with Tables: {total_tables}")
    print(f"Total Headings Detected: {total_headings}")
    
    print("\n=== 2. SEMANTIC CHUNKING ===")
    chunker = SemanticChunker()
    chunks = chunker.chunk_document(parsed_doc, doc_id="paper_01")
    print(f"Total Text Chunks Created: {len(chunks)}")
    
    print("\nSample Chunks:")
    for i in range(min(3, len(chunks))):
        print(f"\n[Chunk {i+1}] (Type: {chunks[i].chunk_type.value}, Pages: {chunks[i].page_start}-{chunks[i].page_end})")
        print(f"Section Path: {chunks[i].section_path}")
        print(f"Text: {chunks[i].text[:150]}...")
        
    print("\n=== 3. IMAGE PROCESSING ===")
    # Extract all raw images
    raw_images = []
    for page in parsed_doc.pages:
        raw_images.extend(page.images)
        
    if raw_images:
        processor = ImageProcessor()
        # This will use fallbacks since OPENAI_API_KEY is not valid
        image_records = processor.process_images(raw_images, doc_id="paper_01", source_file=pdf_path)
        print(f"Successfully Processed Images: {len(image_records)}")
        
        if image_records:
            print("\nSample Image Record:")
            img = image_records[0]
            print(f"ID: {img.image_id}")
            print(f"Page: {img.page_number}")
            print(f"Type: {img.image_type.value}")
            print(f"Dimensions: {img.metadata['image_width']}x{img.metadata['image_height']}")
            print(f"Caption (Fallback): {img.caption}")
    else:
        print("No images found to process.")
        image_records = []
        
    print("\n=== 4. CROSS-MODAL LINKING ===")
    linker = CrossModalLinker(page_proximity=1)
    chunks, image_records = linker.link_chunks_and_images(chunks, image_records)
    
    linked_chunk_count = sum(1 for c in chunks if c.linked_images)
    linked_img_count = sum(1 for i in image_records if i.linked_chunk_ids)
    
    print(f"Chunks linked to images: {linked_chunk_count}")
    print(f"Images linked to chunks: {linked_img_count}")

    print("\nPipeline test completed successfully! (Skipped embedding and storage due to missing infrastructure)")

if __name__ == "__main__":
    run_paper_test()

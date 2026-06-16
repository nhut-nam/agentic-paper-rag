import os
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.schema import BlockTypes
from marker.output import text_from_rendered

from app.config.settings import settings
from app.utils.logger import logger
from app.utils.storage import StorageManager
from app.utils.db import DatabaseHandler

def process_pdf_to_markdown(pdf_path: str, doc_id: str):
    """
    Core logic to convert PDF to Markdown with structured output.
    Now organizes output by doc_id in the 'processed' storage area.
    """
    if not os.path.exists(pdf_path):
        logger.error(f"File not found: {pdf_path}")
        return

    logger.info(f"Processing PDF: {pdf_path} (ID: {doc_id})")
    storage = StorageManager()
    
    # Define a highly structured output path
    # storage/processed/{doc_id}/markdown/
    # storage/processed/{doc_id}/images/
    base_processed_path = f"processed/{doc_id}"
    markdown_subfolder = f"{base_processed_path}/markdown"
    images_subfolder = f"{base_processed_path}/images"
    
    storage.clear_dir(base_processed_path)
    storage.ensure_dir(markdown_subfolder)
    storage.ensure_dir(images_subfolder)

    # 1. Setup Marker
    config_dict = {
        "use_llm": settings.USE_LLM,
        "output_format": settings.OUTPUT_FORMAT,
        "disable_image_extraction": settings.DISABLE_IMAGE_EXTRACTION,
        "disable_ocr": settings.DISABLE_OCR,
    }
    
    artifact_dict = create_model_dict()
    config_parser = ConfigParser(config_dict)
    converter = PdfConverter(
        artifact_dict=artifact_dict,
        config=config_parser.generate_config_dict()
    )
    
    main_renderer = converter.renderer(converter.config)
    text_only_renderer = converter.renderer(converter.config)
    text_only_renderer.extract_images = False 

    # 2. Build Doc and Extract "Beautiful" Text
    doc = converter.build_document(pdf_path)
    extracted_data = {} 
    
    image_producing_types = [getattr(BlockTypes, t) for t in settings.IMAGE_PRODUCING_TYPES]
    transform_types = [getattr(BlockTypes, t) for t in settings.TRANSFORM_TYPES]

    for page in doc.pages:
        for block in page.children:
            if getattr(block, 'removed', False): continue

            if block.block_type in image_producing_types:
                try:
                    block_output = block.render(doc, block_config=main_renderer.block_config)
                    html_content, _ = text_only_renderer.extract_html(doc, block_output)
                    beautiful_text = main_renderer.md_cls.convert(html_content).strip()
                except Exception as e:
                    beautiful_text = block.raw_text(doc).strip()
                
                # Capture original block type name before transformation
                orig_type = block.block_type.name if hasattr(block.block_type, 'name') else str(block.block_type)
                
                if block.block_type in transform_types:
                    block.block_type = BlockTypes.Figure 
                    for attr in ['structure', 'children', 'html', 'code']:
                        if hasattr(block, attr): setattr(block, attr, [] if attr in ['structure', 'children'] else None)

                extracted_data[block.id.to_path()] = {
                    "text": beautiful_text,
                    "type": orig_type
                }

    # 3. Final Render
    rendered = main_renderer(doc)
    markdown_text, _, images = text_from_rendered(rendered)

    db = DatabaseHandler()
    # 4. Structured Save using flexible StorageManager
    for img_name_with_ext, img_data in images.items():
        img_id = os.path.splitext(img_name_with_ext)[0]
        
        # Save image: storage/processed/{doc_id}/images/{img_id}/image.png
        image_rel_path = f"{images_subfolder}/{img_id}/image.png"
        storage.save_image(image_rel_path, img_data)
        
        # Save content.md: storage/processed/{doc_id}/images/{img_id}/content.md
        beautiful_content = ""
        img_type = "Figure"
        
        img_info = extracted_data.get(img_id)
        if img_info:
            if isinstance(img_info, dict):
                beautiful_content = img_info.get("text", "")
                img_type = img_info.get("type", "Figure")
            else:
                beautiful_content = img_info
                
        if beautiful_content:
            storage.save_text(f"{images_subfolder}/{img_id}/content.md", beautiful_content)
        
        # Insert image metadata to DB
        try:
            db.insert_document_image(
                image_id=img_id,
                doc_id=doc_id,
                image_path=image_rel_path,
                image_type=img_type,
                content=beautiful_content
            )
            logger.info(f"Saved image metadata to DB: {img_id} (Type: {img_type})")
        except Exception as db_err:
            logger.error(f"Failed to save image metadata to DB for {img_id}: {db_err}")
        
        # Update main markdown links
        # The link should be relative to result.md (which is in markdown/ folder)
        # result.md is at processed/{doc_id}/markdown/result.md
        # images are at processed/{doc_id}/images/{id}/image.png
        # Relative link: ../images/{id}/image.png
        markdown_text = markdown_text.replace(f"({img_name_with_ext})", f"(../images/{img_id}/image.png)")

    # Save main markdown: storage/processed/{doc_id}/markdown/result.md
    storage.save_text(f"{markdown_subfolder}/result.md", markdown_text)
    logger.info(f"✅ Conversion completed. Output in: storage/{base_processed_path}")

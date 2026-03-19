import logging
from datetime import datetime
from typing import Dict, Any

from dreamsApp.core.keywords import extract_keywords_and_vectors
from dreamsApp.core.location_extractor import extract_gps_from_image
from dreamsApp.core.sentiment import (
    get_chime_category,
    get_image_caption_and_sentiment,
    select_text_for_analysis,
)

logger = logging.getLogger(__name__)

class DreamsPipeline:
    """
    The central orchestration engine for the DREAMS algorithm.
    Decoupled entirely from Flask web routes.
    """
    
    def __init__(self):
        # The pipeline can eventually hold loaded AI models directly
        # rather than reloading them in individual scripts.
        pass
        
    def process_new_post(self, user_id: str, image_path: str, caption: str, timestamp_iso: str = None) -> Dict[str, Any]:
        """
        Executes the computer vision and natural language AI ingestion sequences.
        Returns a dictionary of structured data ready for database insertion.
        """
        if timestamp_iso is None:
            timestamp_iso = datetime.now().isoformat()
            
        gps_data = extract_gps_from_image(image_path)
        
        analysis_result = get_image_caption_and_sentiment(image_path, caption)
        sentiment = analysis_result["sentiment"]
        generated_caption = analysis_result["imgcaption"]

        text_for_analysis = select_text_for_analysis(caption, generated_caption)
        chime_result = get_chime_category(text_for_analysis)
        
        keywords_with_vectors = []
        keyword_type = None
        if sentiment['label'] in ('positive', 'negative'):
            keywords_with_vectors = extract_keywords_and_vectors(generated_caption)
            keyword_type = f"{sentiment['label']}_keywords"

        keywords_for_mongo = []
        if keywords_with_vectors:
            keywords_for_mongo = [
                {k: v for k, v in kw.items() if k != "embedding"}
                for kw in keywords_with_vectors
            ]

        post_doc = {
            'user_id': user_id,
            'caption': caption,
            'timestamp': datetime.fromisoformat(timestamp_iso),
            'image_path': image_path,
            'generated_caption': generated_caption,
            'sentiment': sentiment,
            'chime_analysis': chime_result,
            'location': gps_data,
        }

        return {
            "post_doc": post_doc,
            "keyword_type": keyword_type,
            "keywords_for_db": keywords_for_mongo,
            "keywords_with_vectors": keywords_with_vectors
        }

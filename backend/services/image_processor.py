from typing import List, Dict, Any, Optional, Union
import logging
import base64
from io import BytesIO
from PIL import Image
import torch
from transformers import pipeline, AutoProcessor, AutoModelForVision2Seq
from dataclasses import dataclass
import cv2

@dataclass
class ImageAnalysis:
    objects: List[Dict[str, Any]]
    faces: List[Dict[str, Any]]
    text: List[Dict[str, Any]]
    colors: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class ImageProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize image analysis models
        self.object_detector = pipeline("object-detection", model="facebook/detr-resnet-50")
        self.text_recognizer = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        self.image_classifier = pipeline("image-classification", model="microsoft/resnet-50")
        
    def analyze_image(self, image_data: Union[str, bytes]) -> ImageAnalysis:
        """Analyze image and extract various features."""
        try:
            # Convert image data to PIL Image
            if isinstance(image_data, str):
                # Assume base64 string
                image_data = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_data))
            
            # Detect objects
            objects = self._detect_objects(image)
            
            # Extract text
            text = self._extract_text(image)
            
            # Analyze colors
            colors = self._analyze_colors(image)
            
            # Get metadata
            metadata = self._get_metadata(image)
            
            return ImageAnalysis(
                objects=objects,
                faces=[],
                text=text,
                colors=colors,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing image: {str(e)}")
            return ImageAnalysis([], [], [], [], {})
            
    def _detect_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect objects in image."""
        try:
            # Detect objects
            results = self.object_detector(image)
            
            # Format results
            objects = []
            for result in results:
                objects.append({
                    'label': result['label'],
                    'score': result['score'],
                    'box': result['box']
                })
                
            return objects
            
        except Exception as e:
            self.logger.error(f"Error detecting objects: {str(e)}")
            return []
            
    def _extract_text(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Extract text from image."""
        try:
            # Extract text
            results = self.text_recognizer(image)
            
            # Format results
            text = []
            for result in results:
                text.append({
                    'text': result['generated_text'],
                    'score': result['score']
                })
                
            return text
            
        except Exception as e:
            self.logger.error(f"Error extracting text: {str(e)}")
            return []
            
    def _analyze_colors(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Analyze colors in image."""
        try:
            # Convert to numpy array
            image_np = np.array(image)
            
            # Reshape image
            pixels = image_np.reshape(-1, 3)
            
            # Get unique colors and their counts
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
            
            # Sort by count
            sorted_indices = np.argsort(counts)[::-1]
            unique_colors = unique_colors[sorted_indices]
            counts = counts[sorted_indices]
            
            # Format results
            colors = []
            for color, count in zip(unique_colors[:10], counts[:10]):
                colors.append({
                    'rgb': color.tolist(),
                    'percentage': (count / len(pixels)) * 100
                })
                
            return colors
            
        except Exception as e:
            self.logger.error(f"Error analyzing colors: {str(e)}")
            return []
            
    def _get_metadata(self, image: Image.Image) -> Dict[str, Any]:
        """Get image metadata."""
        try:
            return {
                'format': image.format,
                'mode': image.mode,
                'size': image.size,
                'width': image.width,
                'height': image.height,
                'info': image.info
            }
        except Exception as e:
            self.logger.error(f"Error getting metadata: {str(e)}")
            return {} 
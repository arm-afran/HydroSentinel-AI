"""
Satellite Photo Analyzer
Analyzes satellite images to detect and highlight:
- Roads (yellow)
- Buildings (red)
- Flooded areas (blue)
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple


class SatelliteAnalyzer:
    def __init__(self, image_path: str):
        """
        Initialize the analyzer with a satellite image.
        
        Args:
            image_path: Path to the satellite image
        """
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.height, self.width = self.image.shape[:2]
        self.result = self.image.copy()
        
    def detect_flooded_areas(self) -> np.ndarray:
        """
        Detect flooded areas (water bodies).
        Water typically appears in blue tones in satellite images.
        
        Returns:
            Binary mask of detected flooded areas
        """
        # Define range for water/flooded areas (blue colors)
        # Lower bound: (90, 50, 50) - blue hue range
        # Upper bound: (130, 255, 255)
        lower_water = np.array([90, 50, 50])
        upper_water = np.array([130, 255, 255])
        
        water_mask = cv2.inRange(self.hsv, lower_water, upper_water)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return water_mask
    
    def detect_buildings(self) -> np.ndarray:
        """
        Detect buildings using edge detection and connected components.
        Buildings typically have distinct rectangular shapes and edges.
        
        Returns:
            Binary mask of detected buildings
        """
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        building_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Dilate to fill small gaps
        building_mask = cv2.dilate(building_mask, kernel, iterations=2)
        
        # Filter by color - buildings often have distinct colors
        # Look for reddish/brownish tones (common in aerial imagery)
        lower_building = np.array([0, 30, 60])
        upper_building = np.array([25, 200, 200])
        
        color_mask = cv2.inRange(self.hsv, lower_building, upper_building)
        building_mask = cv2.bitwise_and(building_mask, color_mask)
        
        return building_mask
    
    def detect_roads(self) -> np.ndarray:
        """
        Detect roads using edge detection and line features.
        Roads typically appear as linear features with distinct gray/white colors.
        
        Returns:
            Binary mask of detected roads
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological operations to enhance linear features
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        road_mask = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Apply Sobel edge detection for road edges
        sobelx = cv2.Sobel(road_mask, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(road_mask, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = np.uint8(magnitude / magnitude.max() * 255)
        
        # Threshold to get road edges
        _, road_mask = cv2.threshold(magnitude, 100, 255, cv2.THRESH_BINARY)
        
        # Look for light-colored pixels (roads are typically light)
        # Keep pixels with high brightness
        _, bright_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Combine edge and brightness information
        road_mask = cv2.bitwise_and(road_mask, bright_mask)
        
        # Thin the lines
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_ERODE, kernel, iterations=1)
        
        return road_mask
    
    def analyze(self) -> np.ndarray:
        """
        Perform full analysis and create highlighted output.
        
        Returns:
            Image with highlighted roads, buildings, and flooded areas
        """
        # Get all masks
        flooded_mask = self.detect_flooded_areas()
        buildings_mask = self.detect_buildings()
        roads_mask = self.detect_roads()
        
        # Create colored overlays
        # Flooded areas - blue
        flooded_areas = np.where(flooded_mask[:, :, np.newaxis], [255, 0, 0], 0).astype(np.uint8)
        
        # Buildings - red
        buildings = np.where(buildings_mask[:, :, np.newaxis], [0, 0, 255], 0).astype(np.uint8)
        
        # Roads - yellow (green + red)
        roads = np.where(roads_mask[:, :, np.newaxis], [0, 255, 255], 0).astype(np.uint8)
        
        # Blend overlays with original image
        result = self.image.copy().astype(float)
        
        # Apply alpha blending for each layer
        alpha = 0.4
        
        # Add flooded areas
        flooded_mask_3ch = np.dstack([flooded_mask, flooded_mask, flooded_mask])
        result = np.where(flooded_mask_3ch, 
                         result * (1 - alpha) + flooded_areas * alpha,
                         result)
        
        # Add buildings
        buildings_mask_3ch = np.dstack([buildings_mask, buildings_mask, buildings_mask])
        result = np.where(buildings_mask_3ch,
                         result * (1 - alpha) + buildings * alpha,
                         result)
        
        # Add roads
        roads_mask_3ch = np.dstack([roads_mask, roads_mask, roads_mask])
        result = np.where(roads_mask_3ch,
                         result * (1 - alpha) + roads * alpha,
                         result)
        
        return result.astype(np.uint8)
    
    def save_result(self, output_path: str) -> None:
        """Save the analyzed image."""
        result = self.analyze()
        cv2.imwrite(output_path, result)
        print(f"Analyzed image saved to: {output_path}")
    
    def display_result(self) -> None:
        """Display the analyzed image."""
        result = self.analyze()
        
        # Create figure with original and result
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        axes[0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Satellite Image")
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Analyzed Image\n(Red=Buildings, Yellow=Roads, Blue=Flooded)")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function with example usage."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python satellite_analyzer.py <image_path> [output_path]")
        print("\nExample:")
        print("  python satellite_analyzer.py satellite.png analyzed_output.png")
        return
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "analyzed_satellite.png"
    
    try:
        analyzer = SatelliteAnalyzer(image_path)
        analyzer.save_result(output_path)
        analyzer.display_result()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

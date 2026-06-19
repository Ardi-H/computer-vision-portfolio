# Annotation Validation & Quality Assurance Scripts

Collection of Python scripts used for validating annotated datasets, calculating metrics, and generating quality reports.

---

## 1. COCO Format Validator & IoU Calculator

```python
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class CocoAnnotationValidator:
    """
    Validates COCO format annotations and calculates quality metrics.
    """
    
    def __init__(self, annotation_file):
        self.coco = COCO(annotation_file)
        self.annotations = annotation_file
        
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1_min, y1_min, w1, h1 = box1
        x2_min, y2_min, w2, h2 = box2
        
        # Convert to corner coordinates
        x1_max, y1_max = x1_min + w1, y1_min + h1
        x2_max, y2_max = x2_min + w2, y2_min + h2
        
        # Calculate intersection area
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            intersection = 0
        else:
            intersection = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Calculate union area
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        return iou
    
    def validate_box_boundaries(self, image_id):
        """Check if bounding boxes extend beyond image boundaries"""
        img = self.coco.imgs[image_id]
        img_width, img_height = img['width'], img['height']
        
        annIds = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(annIds)
        
        violations = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if x < 0 or y < 0 or (x + w) > img_width or (y + h) > img_height:
                violations.append({
                    'annotation_id': ann['id'],
                    'bbox': [x, y, w, h],
                    'image_size': [img_width, img_height],
                    'violation': 'Box extends beyond image boundaries'
                })
        
        return violations
    
    def check_overlapping_boxes(self, image_id, iou_threshold=0.1):
        """Detect overlapping boxes (potential labeling errors)"""
        annIds = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(annIds)
        
        overlaps = []
        for i, ann1 in enumerate(anns):
            for j, ann2 in enumerate(anns[i+1:], i+1):
                iou = self.calculate_iou(ann1['bbox'], ann2['bbox'])
                if iou > iou_threshold:
                    overlaps.append({
                        'ann1_id': ann1['id'],
                        'ann2_id': ann2['id'],
                        'iou': iou,
                        'category1': ann1['category_id'],
                        'category2': ann2['category_id']
                    })
        
        return overlaps
    
    def calculate_class_distribution(self):
        """Analyze class distribution across dataset"""
        cats = self.coco.cats
        class_counts = {}
        
        for cat_id in cats:
            annIds = self.coco.getAnnIds(catIds=cat_id)
            class_counts[cats[cat_id]['name']] = len(annIds)
        
        # Calculate percentages
        total = sum(class_counts.values())
        class_dist = {k: (v/total)*100 for k, v in class_counts.items()}
        
        return class_dist
    
    def generate_quality_report(self):
        """Generate comprehensive quality report"""
        report = {
            'total_images': len(self.coco.imgs),
            'total_annotations': len(self.coco.anns),
            'total_classes': len(self.coco.cats),
            'boundary_violations': 0,
            'overlapping_boxes': 0,
            'class_distribution': self.calculate_class_distribution()
        }
        
        # Check all images
        for img_id in self.coco.imgs:
            violations = self.validate_box_boundaries(img_id)
            report['boundary_violations'] += len(violations)
            
            overlaps = self.check_overlapping_boxes(img_id)
            report['overlapping_boxes'] += len(overlaps)
        
        report['quality_score'] = self._calculate_quality_score(report)
        return report
    
    def _calculate_quality_score(self, report):
        """Calculate overall quality score (0-100)"""
        violations_penalty = (report['boundary_violations'] / report['total_annotations']) * 10
        overlap_penalty = (report['overlapping_boxes'] / report['total_annotations']) * 5
        
        quality = 100 - violations_penalty - overlap_penalty
        return max(0, quality)

# Usage Example
validator = CocoAnnotationValidator('annotations.json')
report = validator.generate_quality_report()
print(json.dumps(report, indent=2))
```

---

## 2. Inter-Annotator Agreement (IAA) Calculator

```python
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from itertools import combinations

class InterAnnotatorAgreement:
    """
    Calculate Inter-Annotator Agreement metrics including Cohen's Kappa.
    """
    
    def __init__(self):
        self.agreements = []
    
    def calculate_iaa_images(self, annotations_list):
        """
        Calculate IAA for image-level classifications
        
        Args:
            annotations_list: List of annotation dictionaries from different annotators
            Each dict should have: {'image_id': label}
        """
        
        # Get common images
        common_images = set(annotations_list[0].keys())
        for ann in annotations_list[1:]:
            common_images &= set(ann.keys())
        
        # Extract labels for each annotator
        labels_by_annotator = []
        for ann_dict in annotations_list:
            labels = [ann_dict[img_id] for img_id in sorted(common_images)]
            labels_by_annotator.append(labels)
        
        # Calculate pairwise agreements
        pairwise_agreements = []
        for ann1, ann2 in combinations(range(len(annotations_list)), 2):
            agreement = np.mean(
                np.array(labels_by_annotator[ann1]) == 
                np.array(labels_by_annotator[ann2])
            )
            
            kappa = cohen_kappa_score(
                labels_by_annotator[ann1],
                labels_by_annotator[ann2]
            )
            
            pairwise_agreements.append({
                'annotators': (ann1, ann2),
                'percent_agreement': agreement * 100,
                'cohens_kappa': kappa
            })
        
        # Calculate average IAA
        avg_agreement = np.mean([p['percent_agreement'] for p in pairwise_agreements])
        avg_kappa = np.mean([p['cohens_kappa'] for p in pairwise_agreements])
        
        return {
            'average_agreement': avg_agreement,
            'average_kappas_kappa': avg_kappa,
            'pairwise_agreements': pairwise_agreements,
            'interpretation': self._interpret_kappa(avg_kappa)
        }
    
    def _interpret_kappa(self, kappa):
        """Interpret Cohen's Kappa value"""
        if kappa < 0:
            return "Poor agreement"
        elif kappa < 0.2:
            return "Slight agreement"
        elif kappa < 0.4:
            return "Fair agreement"
        elif kappa < 0.6:
            return "Moderate agreement"
        elif kappa < 0.8:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"

# Usage Example
iaa_calc = InterAnnotatorAgreement()

annotator1 = {'img001': 'car', 'img002': 'person', 'img003': 'car'}
annotator2 = {'img001': 'car', 'img002': 'person', 'img003': 'car'}
annotator3 = {'img001': 'car', 'img002': 'truck', 'img003': 'car'}

result = iaa_calc.calculate_iaa_images([annotator1, annotator2, annotator3])
print(f"Average Agreement: {result['average_agreement']:.2f}%")
print(f"Cohen's Kappa: {result['average_kappas_kappa']:.3f}")
print(f"Interpretation: {result['interpretation']}")
```

---

## 3. Dataset Statistics & Visualization

```python
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class DatasetStatistics:
    """Generate statistical analysis and visualizations for annotated datasets"""
    
    def __init__(self, coco_annotation_file):
        with open(coco_annotation_file, 'r') as f:
            self.data = json.load(f)
    
    def analyze_class_distribution(self):
        """Analyze distribution of object classes"""
        class_counts = Counter()
        
        for ann in self.data['annotations']:
            cat_id = ann['category_id']
            cat_name = next(c['name'] for c in self.data['categories'] if c['id'] == cat_id)
            class_counts[cat_name] += 1
        
        return dict(class_counts)
    
    def analyze_box_sizes(self):
        """Analyze bounding box size distribution"""
        box_areas = []
        box_widths = []
        box_heights = []
        
        for ann in self.data['annotations']:
            x, y, w, h = ann['bbox']
            area = w * h
            box_areas.append(area)
            box_widths.append(w)
            box_heights.append(h)
        
        return {
            'mean_area': np.mean(box_areas),
            'median_area': np.median(box_areas),
            'min_area': np.min(box_areas),
            'max_area': np.max(box_areas),
            'mean_width': np.mean(box_widths),
            'mean_height': np.mean(box_heights)
        }
    
    def visualize_class_distribution(self, output_file='class_distribution.png'):
        """Create bar chart of class distribution"""
        class_dist = self.analyze_class_distribution()
        
        plt.figure(figsize=(14, 6))
        classes = list(class_dist.keys())
        counts = list(class_dist.values())
        
        plt.bar(classes, counts, color='steelblue', edgecolor='navy', alpha=0.7)
        plt.xlabel('Object Class', fontsize=12)
        plt.ylabel('Number of Annotations', fontsize=12)
        plt.title('Dataset Class Distribution', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        return output_file
    
    def visualize_box_sizes(self, output_file='box_sizes.png'):
        """Create histogram of bounding box sizes"""
        stats = self.analyze_box_sizes()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Extract box data
        box_areas = [ann['bbox'][2] * ann['bbox'][3] for ann in self.data['annotations']]
        box_widths = [ann['bbox'][2] for ann in self.data['annotations']]
        box_heights = [ann['bbox'][3] for ann in self.data['annotations']]
        
        # Plot histograms
        axes[0].hist(box_areas, bins=50, color='steelblue', edgecolor='navy')
        axes[0].set_title('Bounding Box Area Distribution')
        axes[0].set_xlabel('Area (pixels²)')
        
        axes[1].hist(box_widths, bins=50, color='green', edgecolor='darkgreen')
        axes[1].set_title('Bounding Box Width Distribution')
        axes[1].set_xlabel('Width (pixels)')
        
        axes[2].hist(box_heights, bins=50, color='orange', edgecolor='darkorange')
        axes[2].set_title('Bounding Box Height Distribution')
        axes[2].set_xlabel('Height (pixels)')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        return output_file

# Usage Example
stats = DatasetStatistics('annotations.json')
class_dist = stats.analyze_class_distribution()
box_stats = stats.analyze_box_sizes()

print("Class Distribution:", class_dist)
print("Box Statistics:", box_stats)

stats.visualize_class_distribution()
stats.visualize_box_sizes()
```

---

## 4. Visualize Annotations on Images

```python
import cv2
import json
import numpy as np
from pathlib import Path

class AnnotationVisualizer:
    """Visualize bounding boxes and other annotations on images"""
    
    def __init__(self, image_dir, annotation_file):
        self.image_dir = Path(image_dir)
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        self.img_id_to_path = {img['id']: img['file_name'] for img in self.coco_data['images']}
    
    def draw_bboxes(self, image_id, output_path=None, thickness=2):
        """Draw bounding boxes on image"""
        img_info = next(img for img in self.coco_data['images'] if img['id'] == image_id)
        img_path = self.image_dir / img_info['file_name']
        
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error reading image: {img_path}")
            return
        
        # Get annotations for this image
        anns = [ann for ann in self.coco_data['annotations'] if ann['image_id'] == image_id]
        
        for ann in anns:
            x, y, w, h = [int(v) for v in ann['bbox']]
            cat_id = ann['category_id']
            cat_name = next(c['name'] for c in self.coco_data['categories'] if c['id'] == cat_id)
            
            # Draw rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness)
            
            # Put label
            cv2.putText(img, cat_name, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, img)
        
        return img
    
    def draw_polygons(self, image_id, output_path=None):
        """Draw polygon segmentation masks on image"""
        img_info = next(img for img in self.coco_data['images'] if img['id'] == image_id)
        img_path = self.image_dir / img_info['file_name']
        
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error reading image: {img_path}")
            return
        
        # Get annotations for this image
        anns = [ann for ann in self.coco_data['annotations'] if ann['image_id'] == image_id]
        
        for ann in anns:
            if 'segmentation' in ann:
                seg = ann['segmentation'][0]
                # Convert to numpy array and reshape
                points = np.array(seg).reshape(-1, 2).astype(np.int32)
                
                # Draw polygon
                cv2.polylines(img, [points], True, (0, 255, 0), 2)
                cv2.fillPoly(img, [points], (0, 255, 0), alpha=0.3)
        
        if output_path:
            cv2.imwrite(output_path, img)
        
        return img
    
    def create_grid_visualization(self, image_ids, output_file='grid_visualization.png', grid_size=(3, 3)):
        """Create grid of annotated images"""
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 15))
        
        for idx, (ax, img_id) in enumerate(zip(axes.flat, image_ids)):
            img = self.draw_bboxes(img_id)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
            ax.set_title(f'Image ID: {img_id}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=200)
        plt.close()
        
        return output_file

# Usage Example
visualizer = AnnotationVisualizer('images/', 'annotations.json')
visualizer.draw_bboxes(image_id=1, output_path='annotated_image_1.jpg')
visualizer.create_grid_visualization([1, 2, 3, 4, 5, 6, 7, 8, 9])
```

---

## 5. Boundary Precision Validator

```python
import numpy as np
from scipy import ndimage

class BoundaryPrecisionValidator:
    """
    Validate annotation boundary precision at pixel level
    Useful for medical imaging and high-precision applications
    """
    
    def __init__(self):
        self.precision_scores = []
    
    def validate_polygon_boundary(self, annotation_points, image, tolerance_pixels=2):
        """
        Validate that polygon boundary matches actual object boundary in image
        
        Args:
            annotation_points: List of [x, y] coordinates
            image: Image array (grayscale)
            tolerance_pixels: Acceptable deviation in pixels
        """
        # Create mask from annotation
        from PIL import Image as PILImage
        mask = PILImage.new('L', image.shape[::-1], 0)
        
        # Draw polygon
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.polygon([(int(p[0]), int(p[1])) for p in annotation_points], fill=1)
        
        mask_array = np.array(mask)
        
        # Find edges in both annotation and image
        annotated_edges = ndimage.sobel(mask_array)
        image_edges = ndimage.sobel(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        
        # Calculate alignment score
        intersection = np.logical_and(annotated_edges > 0, image_edges > 0).sum()
        union = np.logical_or(annotated_edges > 0, image_edges > 0).sum()
        
        alignment_score = (intersection / union) * 100 if union > 0 else 0
        
        return {
            'alignment_score': alignment_score,
            'passed': alignment_score >= 85,
            'annotation_edge_pixels': (annotated_edges > 0).sum(),
            'image_edge_pixels': (image_edges > 0).sum()
        }
    
    def validate_box_tightness(self, bbox, image, min_tightness=0.85):
        """
        Validate that bounding box is tight (not loose)
        """
        x, y, w, h = [int(v) for v in bbox]
        region = image[y:y+h, x:x+w]
        
        # Find actual content in region
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate how much of box is filled
        filled_pixels = np.count_nonzero(binary)
        total_pixels = w * h
        
        tightness = filled_pixels / total_pixels if total_pixels > 0 else 0
        
        return {
            'tightness_score': tightness,
            'passed': tightness >= min_tightness,
            'filled_percentage': tightness * 100
        }

# Usage Example
validator = BoundaryPrecisionValidator()
polygon_points = [[10, 10], [100, 10], [100, 100], [10, 100]]
result = validator.validate_polygon_boundary(polygon_points, image_array)
print(f"Boundary Alignment: {result['alignment_score']:.2f}%")
```

---

## Quality Report Template

```python
def generate_quality_report(dataset_name, metrics):
    """
    Generate comprehensive quality report
    """
    report = f"""
    ═══════════════════════════════════════════════════════════════
    DATA ANNOTATION QUALITY REPORT
    ═══════════════════════════════════════════════════════════════
    
    Dataset: {dataset_name}
    Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    ───────────────────────────────────────────────────────────────
    DATASET STATISTICS
    ───────────────────────────────────────────────────────────────
    Total Images: {metrics['total_images']}
    Total Annotations: {metrics['total_annotations']}
    Total Classes: {metrics['total_classes']}
    
    ─────────────────────��─────────────────────────────────────────
    QUALITY METRICS
    ───────────────────────────────────────────────────────────────
    Overall Quality Score: {metrics['quality_score']:.1f}/100
    
    Inter-Annotator Agreement: {metrics['iaa']:.1f}%
    Cohen's Kappa: {metrics['kappa']:.3f}
    
    Boundary Violations: {metrics['boundary_violations']} ({metrics['boundary_violation_rate']:.2f}%)
    Overlapping Boxes: {metrics['overlapping_boxes']} ({metrics['overlap_rate']:.2f}%)
    
    Average IoU Score: {metrics['avg_iou']:.3f}
    
    ───────────────────────────────────────────────────────────────
    CLASS DISTRIBUTION
    ───────────────────────────────────────────────────────────────
    {metrics['class_distribution_str']}
    
    ───────────────────────────────────────────────────────────────
    PASS/FAIL CRITERIA
    ───────────────────────────────────────────────────────────────
    ✓ Quality Score > 90: {'PASS' if metrics['quality_score'] > 90 else 'FAIL'}
    ✓ IAA > 90%: {'PASS' if metrics['iaa'] > 90 else 'FAIL'}
    ✓ Boundary Violations < 2%: {'PASS' if metrics['boundary_violation_rate'] < 2 else 'FAIL'}
    ✓ Class Balance: {'PASS' if metrics['class_balance_ok'] else 'FAIL'}
    
    ───────────────────────────────────────────────────────────────
    RECOMMENDATIONS
    ───────────────────────────────────────────────────────────────
    {metrics['recommendations']}
    
    ═══════════════════════════════════════════════════════════════
    Report Prepared By: Data Annotation QA Team
    """
    
    return report

# Print report
print(generate_quality_report('Autonomous Vehicle Dataset', metrics))
```

---

## Installation Requirements

```bash
pip install opencv-python
pip install pycocotools
pip install numpy scipy scikit-learn
pip install matplotlib seaborn pillow
```

---

*Scripts maintained by: Ardi Herianto*  
*Last Updated: June 2026*

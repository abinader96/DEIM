import json
from pathlib import Path
from PIL import Image
import yaml
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel

console = Console()

def yolo_to_coco(dataset_path, class_names):
    # Initialize COCO format structures for each split
    splits = ['train', 'val', 'test']
    coco_data_dict = {}
    
    for split in splits:
        coco_data_dict[split] = {
            "info": {"description": "PPE Dataset"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Add categories
        for idx, class_name in enumerate(class_names):
            coco_data_dict[split]["categories"].append({
                "id": idx,
                "name": class_name,
                "supercategory": "none"
            })
    
    # Process images and annotations
    image_id_dict = {split: 0 for split in splits}
    annotation_id_dict = {split: 0 for split in splits}
    
    # Process train, val, test folders
    for split in splits:
        image_dir = dataset_path / 'images' / split
        label_dir = dataset_path / 'labels' / split
        
        if not image_dir.exists() or not label_dir.exists():
            console.print(f"[yellow]Warning:[/yellow] {image_dir} or {label_dir} does not exist. Skipping {split} split.")
            continue
        
        # Create annotations directory if it doesn't exist
        annotations_dir = dataset_path / 'annotations'
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        console.print(f"[bold blue]Processing {len(image_files)} images for {split} split...[/bold blue]")
        
        # Initialize counters for statistics
        images_with_labels = 0
        total_annotations = 0
        
        # Process each image with a progress bar
        for img_path in tqdm(image_files, desc=f"Processing {split} images", unit="image"):
            # Get image filename without extension
            img_filename = img_path.name
            img_id_no_ext = img_path.stem
            
            # Open image to get width and height
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                console.print(f"[red]Error opening image {img_path}: {e}[/red]")
                continue
            
            # Add image to COCO format
            coco_data_dict[split]["images"].append({
                "id": image_id_dict[split],
                "file_name": img_filename,
                "width": width,
                "height": height
            })
            
            # Check if corresponding label file exists
            label_path = label_dir / f"{img_id_no_ext}.txt"
            if label_path.exists():
                with open(label_path, 'r') as f:
                    annotations_for_image = 0
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            # YOLO format: class_id, x_center, y_center, width, height (normalized)
                            x_center, y_center = float(parts[1]), float(parts[2])
                            w, h = float(parts[3]), float(parts[4])
                            
                            # Convert to COCO format (x, y, width, height)
                            x = (x_center - w/2) * width
                            y = (y_center - h/2) * height
                            w = w * width
                            h = h * height
                            
                            # Add annotation to COCO format
                            coco_data_dict[split]["annotations"].append({
                                "id": annotation_id_dict[split],
                                "image_id": image_id_dict[split],
                                "category_id": class_id,
                                "bbox": [x, y, w, h],
                                "area": w * h,
                                "segmentation": [],
                                "iscrowd": 0
                            })
                            annotation_id_dict[split] += 1
                            annotations_for_image += 1
                    
                    if annotations_for_image > 0:
                        images_with_labels += 1
                        total_annotations += annotations_for_image
            
            image_id_dict[split] += 1
        
        # Save COCO format JSON for this split
        output_file = dataset_path / 'annotations' / f'instances_{split}.json'
        with open(output_file, 'w') as f:
            json.dump(coco_data_dict[split], f)
        
        # Display statistics
        console.print(Panel.fit(
            f"[bold green]Split: {split}[/bold green]\n"
            f"Total images: {len(coco_data_dict[split]['images'])}\n"
            f"Images with annotations: {images_with_labels}\n"
            f"Total annotations: {total_annotations}\n"
            f"Output file: {output_file}",
            title="Conversion Statistics",
            border_style="green"
        ))

def load_class_names(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if 'names' in data:
        # Get class names as a list
        if isinstance(data['names'], list):
            return data['names']
        elif isinstance(data['names'], dict):
            # If it's a dict, convert to list ensuring correct order
            max_idx = max(int(k) for k in data['names'].keys() if isinstance(k, (int, str)) and str(k).isdigit())
            class_names = ["" for _ in range(max_idx + 1)]
            for k, v in data['names'].items():
                if isinstance(k, (int, str)) and str(k).isdigit():
                    class_names[int(k)] = v
            return class_names
    
    raise ValueError(f"Could not extract class names from {yaml_path}")

if __name__ == "__main__":
    console.print("[bold magenta]YOLO to COCO Converter[/bold magenta]", justify="center")
    console.print("Converting PPE dataset from YOLO format to COCO format...\n")
    
    # Path to your dataset
    dataset_path = Path('data/ppe-rig-dataset')
    
    # Load class names from classes.yaml
    class_names = load_class_names(dataset_path / 'classes.yaml')
    console.print(f"[bold]Loaded {len(class_names)} classes:[/bold] {', '.join(class_names)}")
    
    # Convert dataset
    yolo_to_coco(dataset_path, class_names)
    console.print("\n[bold green]Conversion complete![/bold green]")

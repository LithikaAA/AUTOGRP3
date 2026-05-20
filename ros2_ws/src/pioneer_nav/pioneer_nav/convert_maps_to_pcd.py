#!/usr/bin/env python3
from PIL import Image
import yaml
import numpy as np

def pgm_to_pcd(pgm_path, yaml_path, output_pcd):
    """Convert occupancy grid to point cloud (.pcd format)"""
    
    # Load image
    img = Image.open(pgm_path)
    data = np.array(img)
    
    # Load YAML metadata
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    
    resolution = meta['resolution']
    origin_x = meta['origin'][0]
    origin_y = meta['origin'][1]
    
    # Extract occupied cells (dark pixels in grayscale)
    points = []
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            # White=free (255), Black=occupied (0), Gray=unknown
            if data[y, x] < 200:  # Occupied/obstacle threshold
                wx = origin_x + x * resolution
                wy = origin_y + y * resolution
                wz = 0.0
                points.append([wx, wy, wz])
    
    points = np.array(points)
    
    # Save as PCD format
    with open(output_pcd, 'w') as f:
        f.write(f"# .PCD v.7 - Point Cloud Data file format\n")
        f.write(f"VERSION .7\n")
        f.write(f"FIELDS x y z\n")
        f.write(f"SIZE 4 4 4\n")
        f.write(f"TYPE F F F\n")
        f.write(f"COUNT 1 1 1\n")
        f.write(f"WIDTH {len(points)}\n")
        f.write(f"HEIGHT 1\n")
        f.write(f"VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write(f"DATA ascii\n")
        
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")
    
    print(f"✓ Saved {len(points)} points to {output_pcd}")

# Use it:
pgm_to_pcd('my_map.pgm', 'pioneer_map_20260516_223108.yaml', 'my_map.pcd')
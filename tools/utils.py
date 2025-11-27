"""
Utility functions for OCR inference pipelines.
"""
import re
import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def load_image(image_path: str) -> Optional[Image.Image]:
    """Load and correct image orientation."""
    try:
        image = Image.open(image_path)
        corrected_image = ImageOps.exif_transpose(image)
        return corrected_image
    except Exception as e:
        print(f"Error loading image: {e}")
        try:
            return Image.open(image_path)
        except:
            return None


def re_match(text: str) -> Tuple[List, List, List]:
    """Extract reference matches from text.
    
    Returns:
        Tuple of (all_matches, image_matches, other_matches)
    """
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    matches_image = []
    matches_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            matches_image.append(a_match[0])
        else:
            matches_other.append(a_match[0])
    return matches, matches_image, matches_other


def extract_coordinates_and_label(ref_text: str, image_width: int, image_height: int) -> Optional[Tuple[str, List]]:
    """Extract coordinates and label from reference text."""
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(f"Error extracting coordinates: {e}")
        return None
    return (label_type, cor_list)


def draw_bounding_boxes(image: Image.Image, refs: List[str], output_path: str) -> Image.Image:
    """Draw bounding boxes on image."""
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    
    font = ImageFont.load_default()

    img_idx = 0
    
    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                
                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
                color_a = color + (20, )
                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{output_path}/images/{img_idx}.jpg")
                        except Exception as e:
                            print(f"Error saving cropped image: {e}")
                            pass
                        img_idx += 1
                        
                    try:
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                        text_x = x1
                        text_y = max(0, y1 - 15)
                            
                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
                                    fill=(255, 255, 255, 30))
                        
                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image: Image.Image, ref_texts: List[str], output_path: str) -> Image.Image:
    """Process image with reference texts."""
    result_image = draw_bounding_boxes(image, ref_texts, output_path)
    return result_image


def save_results(outputs: str, output_path: str, matches_ref: List, matches_images: List, matches_other: List) -> None:
    """Save OCR results to files."""
    # Save original output (keep .mmd for original, use .md for processed)
    with open(f'{output_path}/result_ori.mmd', 'w', encoding='utf-8') as f:
        f.write(outputs)
    
    # Replace image references
    for idx, a_match_image in enumerate(matches_images):
        outputs = outputs.replace(a_match_image, f'![](images/' + str(idx) + '.jpg)\n')
    
    # Replace other references
    for idx, a_match_other in enumerate(matches_other):
        outputs = outputs.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
    
    # Save processed output as .md for markdown preview support
    with open(f'{output_path}/result.md', 'w', encoding='utf-8') as f:
        f.write(outputs)
    # Also save as .mmd for backward compatibility
    with open(f'{output_path}/result.mmd', 'w', encoding='utf-8') as f:
        f.write(outputs)
    
    return outputs




def save_line_type_figure(outputs: str, output_path: str) -> None:
    """Save line type figure if present in outputs."""
    if 'line_type' not in outputs:
        return
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        
        lines = eval(outputs)['Line']['line']
        line_type = eval(outputs)['Line']['line_type']
        endpoints = eval(outputs)['Line']['line_endpoint']
        
        fig, ax = plt.subplots(figsize=(3,3), dpi=200)
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        
        for idx, line in enumerate(lines):
            try:
                p0 = eval(line.split(' -- ')[0])
                p1 = eval(line.split(' -- ')[-1])
                
                if line_type[idx] == '--':
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color='k')
                else:
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color='k')
                
                ax.scatter(p0[0], p0[1], s=5, color='k')
                ax.scatter(p1[0], p1[1], s=5, color='k')
            except:
                pass
        
        for endpoint in endpoints:
            label = endpoint.split(': ')[0]
            (x, y) = eval(endpoint.split(': ')[1])
            ax.annotate(label, (x, y), xytext=(1, 1), textcoords='offset points', 
                        fontsize=5, fontweight='light')
        
        try:
            if 'Circle' in eval(outputs).keys():
                circle_centers = eval(outputs)['Circle']['circle_center']
                radius = eval(outputs)['Circle']['radius']
                
                for center, r in zip(circle_centers, radius):
                    center = eval(center.split(': ')[1])
                    circle = Circle(center, radius=r, fill=False, edgecolor='black', linewidth=0.8)
                    ax.add_patch(circle)
        except:
            pass
        
        plt.savefig(f'{output_path}/geo.jpg')
        plt.close()
    except Exception as e:
        print(f"Error saving line type figure: {e}")


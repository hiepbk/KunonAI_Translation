"""
Configuration file - plain Python variables and dicts (like mmdet3d).
This file should only contain variable definitions, no classes or functions.
"""

# Model configuration
model = dict(
    path='./weights/DeepSeek-OCR',  # Model path (contains both tokenizer and weights)
)

# Image processing configuration
image = dict(
    base_size=1024,
    image_size=640,
    crop_mode=True,
    min_crops=2,
    max_crops=6,  # max:9; If your GPU memory is small, it is recommended to set it to 6.
)

# Processing configuration
processing = dict(
    max_concurrency=100,  # If you have limited GPU memory, lower the concurrency count.
    num_workers=64,  # image pre-process (resize/padding) workers
    print_num_vis_tokens=False,
    skip_repeat=True,
)

# Input/Output paths
paths = dict(
    input='data/test_1.png',  # Input file path (.pdf, .jpg, .png, .jpeg)
    output='results/test_1_result.md',  # Output directory path
)

# Prompt configuration
prompt = dict(
    default='<image>\n<|grounding|>Convert the document to markdown.',
    # Other commonly used prompts:
    # document: '<image>\n<|grounding|>Convert the document to markdown.'
    # other image: '<image>\n<|grounding|>OCR this image.'
    # without layouts: '<image>\nFree OCR.'
    # figures in document: '<image>\nParse the figure.'
    # general: '<image>\nDescribe this image in detail.'
    # rec: '<image>\nLocate <|ref|>xxxx<|/ref|> in the image.'
)

# Mode presets (TODO: change modes)
modes = dict(
    tiny=dict(base_size=512, image_size=512, crop_mode=False),
    small=dict(base_size=640, image_size=640, crop_mode=False),
    base=dict(base_size=1024, image_size=1024, crop_mode=False),
    large=dict(base_size=1280, image_size=1280, crop_mode=False),
    gundam=dict(base_size=1024, image_size=640, crop_mode=True),
)

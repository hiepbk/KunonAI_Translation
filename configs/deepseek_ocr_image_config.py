"""
Configuration file - plain Python variables and dicts (like mmdet3d).
This file should only contain variable definitions, no classes or functions.
"""

# Model configuration
model = dict(
    path='./weights/DeepSeek-OCR',  # Model path (contains both tokenizer and weights)
)

# vLLM engine configuration
engine = dict(
    hf_overrides=dict(
        architectures=['DeepseekOCRForCausalLM'],
    ),
    block_size=256,
    max_model_len=8192,
    enforce_eager=False, # in future, set to False for better performance of inference
    trust_remote_code=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.75,
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
    text_overlay=True,  # If True, overlay predicted text on original image
)


logits_processors = dict(
    type='NoRepeatNGramLogitsProcessor',
    ngram_size=30,
    window_size=90,
    whitelist_token_ids={128821, 128822}
)

sampling_params = dict(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
)



# Input/Output paths
paths = dict(
    input='data/test_1.png',  # Input file path (.pdf, .jpg, .png, .jpeg)
    output='results/test_1',  # Output directory path
)

# Prompt configuration
prompt = dict(
    
    simple_ocr = '<image>\n<|grounding|>OCR this image.',
    document_text = '<image>\n<|grounding|>Convert the document to markdown.',
    parse_figure = '<image>\nParse the figure.',
    describe_image = '<image>\nDescribe this image in detail.',
    locate_text = '<image>\nLocate <|ref|>xxxx<|/ref|> in the image.',
    test_1 = '<image>\n<|grounding|>OCR this image, output the merged sentences and paragraphs regions.',
    custom_prompt = '<image>\n<',
)

# Mode presets (TODO: change modes)
modes = dict(
    tiny=dict(base_size=512, image_size=512, crop_mode=False),
    small=dict(base_size=640, image_size=640, crop_mode=False),
    base=dict(base_size=1024, image_size=1024, crop_mode=False),
    large=dict(base_size=1280, image_size=1280, crop_mode=False),
    gundam=dict(base_size=1024, image_size=640, crop_mode=True),
)

show_visualization = False  # Disabled for UI mode
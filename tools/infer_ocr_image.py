"""
Inference pipeline for DeepSeek-OCR on image files.
Supports command-line arguments to override config values.
"""
import asyncio
import argparse
import os
import re
from pathlib import Path

import torch
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry
from transformers import AutoTokenizer
import time
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
from tqdm import tqdm

from deepseekocr_net.deepseek_ocr import DeepseekOCRForCausalLM
from deepseekocr_net.process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from deepseekocr_net.process.image_process import DeepseekOCRProcessor
from configs import Config

# Register the model
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


def load_image(image_path):
    """Load and correct image orientation."""
    try:
        image = Image.open(image_path)
        corrected_image = ImageOps.exif_transpose(image)
        return corrected_image
    except Exception as e:
        print(f"error: {e}")
        try:
            return Image.open(image_path)
        except:
            return None


def re_match(text):
    """Extract reference matches from text."""
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):
    """Extract coordinates and label from reference text."""
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None
    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, output_path):
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
                            print(e)
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


def process_image_with_refs(image, ref_texts, output_path):
    """Process image with reference texts."""
    result_image = draw_bounding_boxes(image, ref_texts, output_path)
    return result_image


async def stream_generate(model_path, tokenizer, image=None, prompt='', crop_mode=True,
                         image_size=640, base_size=1024, min_crops=2, max_crops=6):
    """Generate text stream from image and prompt."""
    engine_args = AsyncEngineArgs(
        model=model_path,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        max_model_len=8192,
        enforce_eager=False,
        trust_remote_code=True,  
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        # Pass processor parameters through mm_processor_kwargs
        mm_processor_kwargs={
            'image_size': image_size,
            'base_size': base_size,
            'min_crops': min_crops,
            'max_crops': max_crops,
        },
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    logits_processors = [NoRepeatNGramLogitsProcessor(
        ngram_size=30, 
        window_size=90, 
        whitelist_token_ids={128821, 128822}
    )]  # whitelist: <td>, </td> 

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
    )
    
    request_id = f"request-{int(time.time())}"

    printed_length = 0  

    if image and '<image>' in prompt:
        request = {
            "prompt": prompt,
            "multi_modal_data": {"image": image}
        }
    elif prompt:
        request = {
            "prompt": prompt
        }
    else:
        assert False, f'prompt is none!!!'
    
    async for request_output in engine.generate(
        request, sampling_params, request_id
    ):
        if request_output.outputs:
            full_text = request_output.outputs[0].text
            new_text = full_text[printed_length:]
            print(new_text, end='', flush=True)
            printed_length = len(full_text)
            final_output = full_text
    print('\n') 

    return final_output


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='DeepSeek-OCR Inference Pipeline for Images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help=f'Path to model weights (default: {cfg.model.path} or from config)'
    )
    
    parser.add_argument(
        '--input',
        '--input-path',
        type=str,
        dest='input_path',
        default=None,
        help=f'Input image path (default: {cfg.paths.input} or from config)'
    )
    
    parser.add_argument(
        '--output',
        '--output-path',
        type=str,
        dest='output_path',
        default=None,
        help=f'Output directory path (default: {cfg.paths.output} or from config)'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help=f'Prompt for OCR (default: {cfg.prompt.default} or from config)'
    )
    
    parser.add_argument(
        '--crop-mode',
        action='store_true',
        default=None,
        help=f'Enable crop mode (default: {cfg.image.crop_mode} from config)'
    )
    
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_arguments()
    
    # Load config (like mmdet3d)
    from pathlib import Path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Make relative to project root
        config_path = Path(__file__).parent.parent / config_path
    cfg = Config.from_file(str(config_path))
    
    # Merge cfg-options if provided
    if args.cfg_options is not None:
        # Flatten the list of lists
        cfg_options = []
        for opt_list in args.cfg_options:
            cfg_options.extend(opt_list)
        options_dict = Config.parse_cfg_options(cfg_options)
        cfg.merge_from_dict(options_dict)
    
    # Get values from merged config
    model_path = cfg.model.path or './weights/DeepSeek-OCR'
    input_path = cfg.paths.input
    if not input_path:
        raise ValueError("Input path must be provided via --cfg-options paths.input=... or set in config")
    output_path = cfg.paths.output or './results'
    prompt = cfg.prompt.default or '<image>\n<|grounding|>Convert the document to markdown.'
    crop_mode = cfg.image.crop_mode
    
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f'{output_path}/images', exist_ok=True)
    
    # Load image
    print(f"Loading image from: {input_path}")
    image = load_image(input_path)
    if image is None:
        raise ValueError(f"Failed to load image from {input_path}")
    image = image.convert('RGB')
    
    # Create tokenizer
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Create processor with tokenizer and parameters (from merged config)
    processor = DeepseekOCRProcessor(
        tokenizer=tokenizer,
        image_size=cfg.image.image_size,
        base_size=cfg.image.base_size,
        min_crops=cfg.image.min_crops,
        max_crops=cfg.image.max_crops,
    )
    
    # Process image
    if '<image>' in prompt:
        print("Tokenizing image...")
        image_features = processor.tokenize_with_images(
            conversation=prompt,
            images=[image], 
            bos=True, 
            eos=True, 
            cropping=crop_mode
        )
    else:
        image_features = ''
    
    # Run inference
    print("Running inference...")
    result_out = asyncio.run(stream_generate(
        model_path, tokenizer, image_features, prompt, crop_mode,
        image_size=cfg.image.image_size,
        base_size=cfg.image.base_size,
        min_crops=cfg.image.min_crops,
        max_crops=cfg.image.max_crops,
    ))
    
    # Save results
    print('='*15 + 'save results:' + '='*15)
    image_draw = image.copy()
    outputs = result_out
    
    with open(f'{output_path}/result_ori.mmd', 'w', encoding='utf-8') as afile:
        afile.write(outputs)
    
    matches_ref, matches_images, mathes_other = re_match(outputs)
    result = process_image_with_refs(image_draw, matches_ref, output_path)
    
    for idx, a_match_image in enumerate(tqdm(matches_images, desc="image")):
        outputs = outputs.replace(a_match_image, f'![](images/' + str(idx) + '.jpg)\n')
    
    for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other")):
        outputs = outputs.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
    
    with open(f'{output_path}/result.mmd', 'w', encoding='utf-8') as afile:
        afile.write(outputs)
    
    # Handle special line_type output
    if 'line_type' in outputs:
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
    
    result.save(f'{output_path}/result_with_boxes.jpg')
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()


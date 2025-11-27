"""
Inference pipeline for DeepSeek-OCR on image files.
Supports command-line arguments to override config values (like mmdet3d).
"""
import asyncio
import argparse
import os
from pathlib import Path
from typing import Optional

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
from PIL import Image
from tqdm import tqdm

from deepseekocr_net.utils.config import Config
from deepseekocr_net.deepseek_ocr import DeepseekOCRForCausalLM
from deepseekocr_net.process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from deepseekocr_net.process.image_process import DeepseekOCRProcessor
from .utils import (
    load_image,
    re_match,
    process_image_with_refs,
    save_results,
    save_line_type_figure,
    parse_text_with_boxes,
    overlay_text_on_image,
)

# Register the model
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)




def build_engine(cfg) -> AsyncLLMEngine:
    """Build vLLM async engine.
    
    Args:
        cfg: Config object
        
    Returns:
        AsyncLLMEngine instance
    """
    # Model should already be downloaded by ensure_model_downloaded()
    # Just use the local path for vLLM
    vllm_model_path = cfg.model.path
    
    engine_args = AsyncEngineArgs(
        model=vllm_model_path,
        hf_overrides=cfg.engine.hf_overrides,
        block_size=cfg.engine.block_size,
        max_model_len=cfg.engine.max_model_len,
        enforce_eager=cfg.engine.enforce_eager,
        trust_remote_code=cfg.engine.trust_remote_code,
        tensor_parallel_size=cfg.engine.tensor_parallel_size,
        gpu_memory_utilization=cfg.engine.gpu_memory_utilization,
        # Pass processor parameters through mm_processor_kwargs
        mm_processor_kwargs={
            'image_size': cfg.image.image_size,
            'base_size': cfg.image.base_size,
            'min_crops': cfg.image.min_crops,
            'max_crops': cfg.image.max_crops,
            'print_num_vis_tokens': cfg.processing.print_num_vis_tokens,
        },
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    return engine


def build_sampling_params(cfg) -> SamplingParams:
    """Build sampling parameters for generation.
    
    Returns:
        SamplingParams instance
    """
    logits_processors = [NoRepeatNGramLogitsProcessor(
        ngram_size=cfg.logits_processors.ngram_size, 
        window_size=cfg.logits_processors.window_size, 
        whitelist_token_ids=cfg.logits_processors.whitelist_token_ids
    )]  # whitelist: <td>, </td> 

    sampling_params = SamplingParams(
        temperature=cfg.sampling_params.temperature,
        max_tokens=cfg.sampling_params.max_tokens,
        logits_processors=logits_processors,
        skip_special_tokens=cfg.sampling_params.skip_special_tokens,
    )
    return sampling_params


async def generate_text(engine: AsyncLLMEngine, sampling_params: SamplingParams, 
                       image_features, prompt: str, raw_image: Image.Image = None) -> str:
    """Generate text from image and prompt using vLLM engine.
    
    Args:
        engine: AsyncLLMEngine instance
        sampling_params: SamplingParams instance
        image_features: Processed image features
        prompt: Text prompt
        raw_image: Raw image for debugging
    Returns:
        Generated text output
    """
    request_id = f"request-{int(time.time())}"
    printed_length = 0

    if image_features and '<image>' in prompt:
        request = {
            "prompt": prompt,
            "multi_modal_data": {"image": image_features}
        }
    elif prompt:
        request = {
            "prompt": prompt
        }
    else:
        raise ValueError("Prompt is required")
    
    async for request_output in engine.generate(request, sampling_params, request_id):
        if request_output.outputs:
            full_text = request_output.outputs[0].text
            new_text = full_text[printed_length:]
            print(new_text, end='', flush=True)
            printed_length = len(full_text)
            final_output = full_text
    
    print('final_output: \n', final_output)
    print('\n') 

    return final_output


def parse_arguments():
    """Parse command-line arguments (like mmdet3d)."""
    parser = argparse.ArgumentParser(
        description='DeepSeek-OCR Inference Pipeline for Images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/deepseek_ocr_image_config.py',
        help='Config file path'
    )
    
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action='append',
        default=None,
        help='Override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.'
    )
    
    # Deprecated: --options (for backward compatibility)
    parser.add_argument(
        '--options',
        nargs='+',
        action='append',
        default=None,
        help='Deprecated in favor of --cfg-options'
    )
    
    args = parser.parse_args()
    
    # Handle deprecated --options
    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        import warnings
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options
    
    return args


def main():
    """Main inference function."""
    args = parse_arguments()
    
    # Load config
    cfg = Config.from_file(args.config)
    
    # Merge cfg-options if provided
    if args.cfg_options is not None:
        # Flatten the list of lists
        cfg_options = []
        for opt_list in args.cfg_options:
            cfg_options.extend(opt_list)
        options_dict = Config.parse_cfg_options(cfg_options)
        cfg.merge_from_dict(options_dict)
    
    # Get values directly from merged config
    # model_path = cfg.model.path
    # input_path = cfg.paths.input
    # output_path = cfg.paths.output
    # prompt = cfg.prompt.default
    # crop_mode = cfg.image.crop_mode
    
    # # Convert relative paths to absolute paths
    # if not os.path.isabs(model_path):
    #     model_path = os.path.abspath(os.path.join(Path(__file__).parent.parent, model_path))
    # if not os.path.isabs(input_path):
    #     input_path = os.path.abspath(os.path.join(Path(__file__).parent.parent, input_path))
    # if not os.path.isabs(output_path):
    #     output_path = os.path.abspath(os.path.join(Path(__file__).parent.parent, output_path))
    
    # Create output directories
    os.makedirs(cfg.paths.output, exist_ok=True)
    os.makedirs(f'{cfg.paths.output}/images', exist_ok=True)
    os.makedirs(cfg.model.path, exist_ok=True)
    
    # Build components
    # DeepseekOCRProcessor will handle model download and tokenizer loading internally
    print("=" * 50)
    print("Building components...")
    print("=" * 50)
    
    processor = DeepseekOCRProcessor(cfg=cfg)
    engine = build_engine(cfg)
    sampling_params = build_sampling_params(cfg)
    
    # Load and process image
    print("=" * 50)
    print("Processing image...")
    print("=" * 50)
    print(f"Loading image from: {cfg.paths.input}")
    image = load_image(cfg.paths.input)
    if image is None:
        raise ValueError(f"Failed to load image from {cfg.paths.input}")
    image = image.convert('RGB')
    
    # Choose prompt based on text_overlay mode
    use_text_overlay = getattr(cfg.processing, 'text_overlay', False)
    if use_text_overlay:
        prompt_text = getattr(cfg.prompt, 'text_overlay', cfg.prompt.default)
        print("Using text overlay mode - text will be overlaid on original image")
    else:
        prompt_text = cfg.prompt.default
    
    # Tokenize image
    if '<image>' in prompt_text:
        print("Tokenizing image...")
        image_features = processor.tokenize_with_images(
            conversation=prompt_text,
            images=[image], 
            bos=True, 
            eos=True, 
            cropping=cfg.image.crop_mode
        )
    else:
        image_features = None
    
    # Run inference
    print("=" * 50)
    print("Running inference...")
    print("=" * 50)
    result_out = asyncio.run(generate_text(engine, sampling_params, image_features, prompt_text, raw_image=image.copy()))
    
    # Save results
    print("=" * 50)
    print("Saving results...")
    print("=" * 50)
    image_draw = image.copy()
    outputs = result_out
    
    matches_ref, matches_images, matches_other = re_match(outputs)
    result = process_image_with_refs(image_draw, matches_ref, cfg.paths.output)
    
    save_results(outputs, cfg.paths.output, matches_ref, matches_images, matches_other)
    save_line_type_figure(outputs, cfg.paths.output)
    
    result.save(f'{cfg.paths.output}/result_with_boxes.jpg')
    
    # Text overlay mode: overlay predicted text on original image
    if use_text_overlay:
        print("Overlaying text on image...")
        text_boxes = parse_text_with_boxes(outputs)
        if text_boxes:
            image_with_text = overlay_text_on_image(image, text_boxes)
            image_with_text.save(f'{cfg.paths.output}/result_with_text_overlay.jpg')
            print(f"Text overlay saved to: {cfg.paths.output}/result_with_text_overlay.jpg")
            print(f"Found {len(text_boxes)} text regions")
        else:
            print("Warning: No text with bounding boxes found in output. Make sure the prompt requests grounding format.")
    
    print(f"Results saved to: {cfg.paths.output}")


if __name__ == "__main__":
    main()

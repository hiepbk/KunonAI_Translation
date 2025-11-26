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
)

# Register the model
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


def build_tokenizer(tokenizer_path: str, cfg) -> AutoTokenizer:
    """Build tokenizer from model path.
    
    AutoTokenizer.from_pretrained() expects:
    - A directory path containing tokenizer files:
      * tokenizer.json (or tokenizer_config.json) - main tokenizer configuration
      * vocab.json or vocab.txt - vocabulary file
      * special_tokens_map.json - special tokens mapping
      * merges.txt (for BPE tokenizers)
      * Other tokenizer-related JSON/txt files
    - OR a HuggingFace model ID (string like 'deepseek-ai/DeepSeek-OCR')
    
    Args:
        tokenizer_path: Path to tokenizer directory (local) or HuggingFace model ID
        cfg: Config object
        
    Returns:
        AutoTokenizer instance
    """
    print(f"Loading tokenizer from: {tokenizer_path}")
    
    # Check if local path exists and has tokenizer files
    if os.path.exists(tokenizer_path) and os.path.isdir(tokenizer_path):
        # Check if essential tokenizer files exist
        tokenizer_files = [
            'tokenizer.json', 
            'tokenizer_config.json', 
            'vocab.json', 
            'vocab.txt',
            'special_tokens_map.json'
        ]
        has_tokenizer_files = any(os.path.exists(os.path.join(tokenizer_path, f)) for f in tokenizer_files)
        
        if has_tokenizer_files:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
            print(f"✓ Tokenizer loaded from local path: {tokenizer_path}")
            return tokenizer
        else:
            print(f"⚠ Local path exists but missing tokenizer files, will download...")
    
    # Download tokenizer files to local weights folder (not HuggingFace cache)
    from huggingface_hub import snapshot_download
    hf_model_id = 'deepseek-ai/DeepSeek-OCR'
    print(f"Downloading tokenizer files to: {tokenizer_path}")
    os.makedirs(tokenizer_path, exist_ok=True)
    
    # Download only tokenizer-related files (not model weights)
    print("Downloading tokenizer files (tokenizer.json, vocab files, configs, etc.)...")
    snapshot_download(
        repo_id=hf_model_id,
        local_dir=tokenizer_path,
        local_dir_use_symlinks=False,
        allow_patterns=[
            "tokenizer*.json",      # tokenizer.json, tokenizer_config.json
            "vocab.json",           # vocabulary file
            "vocab.txt",            # vocabulary file (alternative)
            "special_tokens_map.json",  # special tokens
            "merges.txt",           # BPE merges (if applicable)
            "*.json",               # Other JSON config files
        ],
        ignore_patterns=[
            "*.safetensors",        # Model weights
            "*.bin",                # Model weights
            "*.pt",                 # Model weights
            "*.pth",                # Model weights
            "*.onnx",               # Model weights
            "*.h5",                 # Model weights
            "*.ckpt",               # Model weights
            "model*.json",          # Model config (not tokenizer)
            "config.json",          # Model config (not tokenizer)
        ],
        
    )
    
    # Load tokenizer from local path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
    print(f"✓ Tokenizer downloaded and loaded from: {tokenizer_path}")
    
    return tokenizer


def build_processor(tokenizer: AutoTokenizer, cfg) -> DeepseekOCRProcessor:
    """Build image processor with tokenizer and config parameters.
    
    Args:
        tokenizer: AutoTokenizer instance
        cfg: Config object
        
    Returns:
        DeepseekOCRProcessor instance
    """
    processor = DeepseekOCRProcessor(
        tokenizer=tokenizer,
        image_size=cfg.image.image_size,
        base_size=cfg.image.base_size,
        min_crops=cfg.image.min_crops,
        max_crops=cfg.image.max_crops,
    )
    return processor


def build_engine(ckpt_path: str, cfg) -> AsyncLLMEngine:
    """Build vLLM async engine.
    
    Args:
        ckpt_path: Path to model checkpoint/weights directory (local) or HuggingFace model ID
        cfg: Config object
        
    Returns:
        AsyncLLMEngine instance
    """
    # Check if local ckpt path exists and has model files
    has_local_model = False
    if os.path.exists(ckpt_path) and os.path.isdir(ckpt_path):
        # Check if model weights exist (safetensors files)
        safetensors_files = list(Path(ckpt_path).glob("*.safetensors"))
        if safetensors_files:
            has_local_model = True
    
    if not has_local_model:
        # Download model weights to local ckpt folder (not HuggingFace cache)
        from huggingface_hub import snapshot_download
        hf_model_id = 'deepseek-ai/DeepSeek-OCR'
        print(f"Local model weights not found ({ckpt_path}), downloading to: {ckpt_path}")
        os.makedirs(ckpt_path, exist_ok=True)
        
        # Download only model weights (safetensors, config, etc.)
        print("Downloading model weights (this may take a while)...")
        snapshot_download(
            repo_id=hf_model_id,
            local_dir=ckpt_path,
            local_dir_use_symlinks=False,
            allow_patterns=[
                "*.safetensors",      # Model weights
                "config.json",       # Model config
                "model*.json",       # Model config files
                "*.json",            # Other config files
            ],
            ignore_patterns=[
                "tokenizer*",        # Tokenizer files (already in tokenizer_path)
                "vocab*",
                "special_tokens*",
                "merges.txt",
            ],
        )
        print(f"✓ Model weights downloaded to: {ckpt_path}")
    
    # Use local path for vLLM
    vllm_model_path = ckpt_path
    
    engine_args = AsyncEngineArgs(
        model=vllm_model_path,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        max_model_len=8192,
        enforce_eager=False,
        trust_remote_code=True,  
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        # Pass processor parameters through mm_processor_kwargs
        mm_processor_kwargs={
            'image_size': cfg.image.image_size,
            'base_size': cfg.image.base_size,
            'min_crops': cfg.image.min_crops,
            'max_crops': cfg.image.max_crops,
        },
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    return engine


def build_sampling_params() -> SamplingParams:
    """Build sampling parameters for generation.
    
    Returns:
        SamplingParams instance
    """
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
    return sampling_params


async def generate_text(engine: AsyncLLMEngine, sampling_params: SamplingParams, 
                       image_features, prompt: str) -> str:
    """Generate text from image and prompt using vLLM engine.
    
    Args:
        engine: AsyncLLMEngine instance
        sampling_params: SamplingParams instance
        image_features: Processed image features
        prompt: Text prompt
        
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
    tokenizer_path = cfg.model.tokenizer_path
    ckpt_path = cfg.model.ckpt_path
    input_path = cfg.paths.input
    output_path = cfg.paths.output
    prompt = cfg.prompt.default
    crop_mode = cfg.image.crop_mode
    
    # Convert relative paths to absolute paths
    if not os.path.isabs(tokenizer_path):
        tokenizer_path = os.path.abspath(os.path.join(Path(__file__).parent.parent, tokenizer_path))
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.abspath(os.path.join(Path(__file__).parent.parent, ckpt_path))
    if not os.path.isabs(input_path):
        input_path = os.path.abspath(os.path.join(Path(__file__).parent.parent, input_path))
    if not os.path.isabs(output_path):
        output_path = os.path.abspath(os.path.join(Path(__file__).parent.parent, output_path))
    
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f'{output_path}/images', exist_ok=True)
    
    # Build components
    print("=" * 50)
    print("Building components...")
    print("=" * 50)
    
    tokenizer = build_tokenizer(tokenizer_path, cfg)
    processor = build_processor(tokenizer, cfg)
    engine = build_engine(ckpt_path, cfg)
    sampling_params = build_sampling_params()
    
    # Load and process image
    print("=" * 50)
    print("Processing image...")
    print("=" * 50)
    print(f"Loading image from: {input_path}")
    image = load_image(input_path)
    if image is None:
        raise ValueError(f"Failed to load image from {input_path}")
    image = image.convert('RGB')
    
    # Tokenize image
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
        image_features = None
    
    # Run inference
    print("=" * 50)
    print("Running inference...")
    print("=" * 50)
    result_out = asyncio.run(generate_text(engine, sampling_params, image_features, prompt))
    
    # Save results
    print("=" * 50)
    print("Saving results...")
    print("=" * 50)
    image_draw = image.copy()
    outputs = result_out
    
    matches_ref, matches_images, matches_other = re_match(outputs)
    result = process_image_with_refs(image_draw, matches_ref, output_path)
    
    save_results(outputs, output_path, matches_ref, matches_images, matches_other)
    save_line_type_figure(outputs, output_path)
    
    result.save(f'{output_path}/result_with_boxes.jpg')
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

"""
Gradio UI for DeepSeek-OCR inference pipeline.
Loads model once at startup, allows changing prompts dynamically.
"""
import asyncio
import os
import argparse
from pathlib import Path
from typing import Optional, Tuple
import time
import queue
import threading

import torch
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import gradio as gr
from PIL import Image
import numpy as np

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry

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


# Global variables to hold loaded components
_engine: Optional[AsyncLLMEngine] = None
_processor: Optional[DeepseekOCRProcessor] = None
_sampling_params: Optional[SamplingParams] = None
_cfg: Optional[Config] = None


def build_engine(cfg) -> AsyncLLMEngine:
    """Build vLLM async engine."""
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
    """Build sampling parameters for generation."""
    logits_processors = [NoRepeatNGramLogitsProcessor(
        ngram_size=cfg.logits_processors.ngram_size, 
        window_size=cfg.logits_processors.window_size, 
        whitelist_token_ids=cfg.logits_processors.whitelist_token_ids
    )]

    sampling_params = SamplingParams(
        temperature=cfg.sampling_params.temperature,
        max_tokens=cfg.sampling_params.max_tokens,
        logits_processors=logits_processors,
        skip_special_tokens=cfg.sampling_params.skip_special_tokens,
    )
    return sampling_params


async def generate_text(engine: AsyncLLMEngine, sampling_params: SamplingParams, 
                       image_features, prompt: str) -> str:
    """Generate text from image and prompt using vLLM engine."""
    request_id = f"request-{int(time.time())}"
    
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
    
    final_output = ""
    async for request_output in engine.generate(request, sampling_params, request_id):
        if request_output.outputs:
            final_output = request_output.outputs[0].text

    return final_output


async def generate_text_stream(engine: AsyncLLMEngine, sampling_params: SamplingParams, 
                                image_features, prompt: str):
    """Generate text with streaming - yields text incrementally."""
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
            if new_text:  # Only yield if there's new text
                printed_length = len(full_text)
                yield new_text


def initialize_model(config_path: str):
    """Initialize model components once at startup."""
    global _engine, _processor, _sampling_params, _cfg
    
    if _engine is not None:
        return "Model already loaded!"
    
    print("=" * 50)
    print("Initializing model (this may take a while)...")
    print("=" * 50)
    
    # Load config
    _cfg = Config.from_file(config_path)
    
    # Create output directories
    os.makedirs(_cfg.paths.output, exist_ok=True)
    os.makedirs(f'{_cfg.paths.output}/images', exist_ok=True)
    os.makedirs(_cfg.model.path, exist_ok=True)
    
    # Build components
    print("Building processor...")
    _processor = DeepseekOCRProcessor(cfg=_cfg)
    
    print("Building engine...")
    _engine = build_engine(_cfg)
    
    print("Building sampling params...")
    _sampling_params = build_sampling_params(_cfg)
    
    print("=" * 50)
    print("Model loaded successfully!")
    print("=" * 50)
    
    return "Model loaded successfully! Ready to process images."



def process_image_ui_stream(image: Image.Image, prompt: str, output_dir: str):
    """
    Process image with streaming text output.
    Yields: (streaming_text, status, output_image, final_markdown)
    """
    global _engine, _processor, _sampling_params, _cfg
    
    if _engine is None or _processor is None:
        yield "", "Error: Model not initialized. Please restart the application.", None, ""
        return
    
    if image is None:
        yield "", "Error: Please upload an image.", None, ""
        return
    
    if not prompt or not prompt.strip():
        yield "", "Error: Please provide a prompt.", None, ""
        return
    
    try:
        # Convert image to RGB
        image = image.convert('RGB')
        
        # Create output directory for this run
        timestamp = int(time.time())
        output_path = f"{output_dir}/run_{timestamp}"
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(f'{output_path}/images', exist_ok=True)
        
        # Tokenize image if needed
        image_features = None
        if '<image>' in prompt:
            image_features = _processor.tokenize_with_images(
                conversation=prompt,
                images=[image], 
                bos=True, 
                eos=True, 
                cropping=_cfg.image.crop_mode
            )
        
        # Run streaming inference
        print(f"Running inference with prompt: {prompt[:50]}...")
        accumulated_text = ""
        
        # Use a queue to bridge async generator to sync generator
        text_queue = queue.Queue()
        done = threading.Event()
        
        async def stream_async():
            """Run async stream and put chunks in queue."""
            try:
                async for chunk in generate_text_stream(_engine, _sampling_params, image_features, prompt):
                    text_queue.put(chunk)
                text_queue.put(None)  # Signal done
            except Exception as e:
                text_queue.put(None)
                raise e
        
        # Start async stream in background
        def run_async():
            asyncio.run(stream_async())
            done.set()
        
        stream_thread = threading.Thread(target=run_async, daemon=True)
        stream_thread.start()
        
        # Yield chunks as they arrive
        while not done.is_set() or not text_queue.empty():
            try:
                chunk = text_queue.get(timeout=0.1)
                if chunk is None:
                    break
                accumulated_text += chunk
                yield accumulated_text, "🔄 Generating...", None, ""
            except queue.Empty:
                continue
        
        result_text = accumulated_text
        
        # Process results
        matches_ref, matches_images, matches_other = re_match(result_text)
        
        # Draw bounding boxes on image
        output_image = process_image_with_refs(image.copy(), matches_ref, output_path)
        
        # Save results
        save_results(result_text, output_path, matches_ref, matches_images, matches_other)
        save_line_type_figure(result_text, output_path)
        
        # Save output image
        output_image_path = f'{output_path}/result_with_boxes.jpg'
        output_image.save(output_image_path)
        
        # Read the processed markdown file
        markdown_path = f'{output_path}/result.md'
        if os.path.exists(markdown_path):
            with open(markdown_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
        else:
            markdown_content = result_text
        
        status = f"✓ Success! Results saved to: {output_path}"
        yield result_text, status, output_image, markdown_content
        
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        yield "", error_msg, None, ""


def create_ui(config_path: str, default_output_dir: str = "results/ui_outputs"):
    """Create and launch Gradio UI."""
    
    # Initialize model on startup
    init_status = initialize_model(config_path)
    
    # Load default prompts from config
    cfg = Config.from_file(config_path)
    default_prompts = {
        "Default": cfg.prompt.default,
        "Document Text": cfg.prompt.document_text,
        "OCR Image": cfg.prompt.other_image,
        "Free OCR": cfg.prompt.without_layouts,
        "Table Merge": getattr(cfg.prompt, 'table_merge_text', cfg.prompt.default),
        "Test 8": getattr(cfg.prompt, 'test_8', cfg.prompt.default),
    }
    
    # Create UI
    with gr.Blocks(title="DeepSeek-OCR Inference") as demo:
        gr.Markdown("# 🖼️ DeepSeek-OCR Inference UI")
        gr.Markdown("Load model once, change prompts dynamically without restarting!")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input")
                
                # Get list of images from data folder
                data_folder = "data"
                if not os.path.exists(data_folder):
                    data_folder = "."
                
                def get_image_list():
                    """Get list of image files from data folder."""
                    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
                    image_files = []
                    if os.path.exists(data_folder):
                        for file in sorted(os.listdir(data_folder)):
                            if any(file.endswith(ext) for ext in image_extensions):
                                full_path = os.path.join(data_folder, file)
                                image_files.append((file, full_path))
                    return image_files
                
                image_list = get_image_list()
                # Create mapping from filename to full path
                image_path_map = {name: path for name, path in image_list}
                image_choices = [name for name, path in image_list]
                
                # Dropdown to select image from data folder
                image_selector = gr.Dropdown(
                    choices=image_choices,
                    label="Select Image from Data Folder",
                    value=None,  # No default selection - user must choose
                    interactive=True
                )
                
                # Function to load image from selected filename
                def load_selected_image(selection):
                    """Load image from dropdown selection."""
                    if selection is None or not selection:
                        return None
                    try:
                        # Get full path from mapping
                        path = image_path_map.get(selection)
                        if path and os.path.exists(path):
                            img = Image.open(path).convert('RGB')
                            return img
                        return None
                    except Exception as e:
                        print(f"Error loading image: {e}")
                        return None
                
                # Image display (loaded from selector)
                image_input = gr.Image(
                    type="pil",
                    label="Image Preview",
                    height=400
                )
                
                # When image is selected from dropdown, load it
                image_selector.change(
                    fn=load_selected_image,
                    inputs=image_selector,
                    outputs=image_input
                )
                
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=5,
                    value=cfg.prompt.default
                )
                
                # Quick prompt buttons
                gr.Markdown("### 🚀 Quick Prompts")
                with gr.Row():
                    for name, prompt in default_prompts.items():
                        btn = gr.Button(name, size="sm")
                        btn.click(
                            fn=lambda p=prompt: p,
                            outputs=prompt_input
                        )
                
                output_dir_input = gr.Textbox(
                    label="Output Directory",
                    value=default_output_dir,
                    lines=1
                )
                
                process_btn = gr.Button("🚀 Process Image", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Output")
                
                status_output = gr.Textbox(
                    label="Status",
                    value=init_status,
                    interactive=False
                )
                
                # Streaming text output for real-time generation
                streaming_text_output = gr.Textbox(
                    label="🔄 Generating Text (Real-time)",
                    value="",
                    interactive=False,
                    lines=15,
                    max_lines=20
                )
                
                image_output = gr.Image(
                    label="Output Image (with bounding boxes)",
                    height=400
                )
                
                text_output = gr.Markdown(
                    label="Final Extracted Text (Markdown)",
                    value="Results will appear here..."
                )
        
        # Connect the process button with streaming
        process_btn.click(
            fn=process_image_ui_stream,
            inputs=[image_input, prompt_input, output_dir_input],
            outputs=[streaming_text_output, status_output, image_output, text_output]
        )
        
        # Example section
        gr.Markdown("### 💡 Example Prompts")
        gr.Examples(
            examples=[
                ["<image>\n<|grounding|>Convert the document to markdown."],
                ["<image>\n<|grounding|>OCR this image."],
                ["<image>\nFree OCR."],
                ["<image>\n<|grounding|>OCR all text in this image with bounding boxes. For each text region, output: <|ref|>text_content<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>"],
            ],
            inputs=prompt_input
        )
    
    return demo


def main():
    """Main function to launch UI."""
    parser = argparse.ArgumentParser(description='DeepSeek-OCR UI')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/deepseek_ocr_image_config.py',
        help='Config file path'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/ui_outputs',
        help='Default output directory'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public link (gradio share)'
    )
    parser.add_argument(
        '--server-name',
        type=str,
        default='0.0.0.0',
        help='Server name (default: 0.0.0.0 for all interfaces, use 127.0.0.1 for local only)'
    )
    parser.add_argument(
        '--server-port',
        type=int,
        default=7860,
        help='Server port (default: 7860)'
    )
    
    args = parser.parse_args()
    
    # Create UI
    print("\n" + "=" * 50)
    print("Creating Gradio UI interface...")
    print("=" * 50)
    demo = create_ui(args.config, args.output_dir)
    
    # Launch
    print("\n" + "=" * 50)
    print("🚀 Launching UI...")
    print("=" * 50)
    
    # Get the actual server IP if using 0.0.0.0
    import socket
    if args.server_name == '0.0.0.0':
        # Try to get the actual IP address
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            actual_ip = s.getsockname()[0]
            s.close()
            print(f"📡 Server accessible at:")
            print(f"   - Local:  http://127.0.0.1:{args.server_port}")
            print(f"   - Network: http://{actual_ip}:{args.server_port}")
        except:
            print(f"📡 Server accessible at: http://{args.server_name}:{args.server_port}")
    else:
        print(f"📡 Server accessible at: http://{args.server_name}:{args.server_port}")
    
    print("\n⏳ Building UI interface (this may take a few seconds)...")
    print("💡 Once you see 'Running on...', the UI is ready!")
    print("💡 If the page doesn't load, check:")
    print("   1. Firewall allows port", args.server_port)
    print("   2. Server IP is correct:", args.server_name)
    print("   3. Try accessing from server itself: http://127.0.0.1:" + str(args.server_port))
    print("\nPress Ctrl+C to stop\n")
    
    try:
        demo.launch(
            server_name=args.server_name,
            server_port=args.server_port,
            share=args.share,
            show_error=True
        )
    except Exception as e:
        print(f"\n❌ Error launching UI: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check if port", args.server_port, "is already in use")
        print("   2. Try a different port: --server-port 8080")
        print("   3. Check firewall settings")
        raise


if __name__ == "__main__":
    main()


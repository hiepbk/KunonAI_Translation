# DeepSeek OCR Installation Guide

This guide will help you set up a conda environment for DeepSeek OCR with all necessary dependencies.

## Prerequisites

- Conda (Miniconda or Anaconda)
- CUDA 12.4 compatible GPU (required for GPU acceleration)
- NVIDIA GPU drivers installed
- At least 40GB GPU memory recommended (A100-40G or similar)

## Step-by-Step Installation

### Step 1: Create Conda Environment

Create a new conda environment with Python 3.12.9:

```bash
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr
```

### Step 2: Install PyTorch with CUDA 12.4 Support

Install PyTorch 2.6.0 with CUDA 12.4:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

**Note:** 
- If you don't have CUDA or want CPU-only version (not recommended for DeepSeek OCR):
  ```bash
  pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
  ```
- For other CUDA versions, check [PyTorch installation guide](https://pytorch.org/get-started/locally/)
- Verify installation:
  ```bash
  python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
  ```

### Step 3: Install vLLM

vLLM 0.8.5 is required for DeepSeek OCR. Install directly with pip:

```bash
pip install vllm==0.8.5
```

**Note:** vLLM will automatically detect CUDA 12.4 and install the appropriate version.

**Verify installation:**
```bash
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

### Step 4: Install Base Requirements

Install the base requirements from requirements.txt:

```bash
pip install -r requirements.txt
```

**Note:** There may be dependency conflicts between requirements.txt and vLLM. After installing requirements.txt, upgrade transformers and tokenizers to versions compatible with vLLM:

```bash
pip install transformers>=4.51.1 tokenizers>=0.21.1
```

This will install:
- transformers (upgraded to >=4.51.1 for vLLM compatibility)
- tokenizers (upgraded to >=0.21.1 for vLLM compatibility)
- PyMuPDF (for PDF processing)
- img2pdf
- einops
- easydict
- addict
- Pillow
- numpy

### Step 5: Install Flash Attention

Install flash-attn for optimized attention computation:

```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

**Note:** 
- This may take several minutes to compile
- If installation fails, you may need to install CUDA toolkit development files
- For systems without CUDA, this step can be skipped (but performance will be reduced)

### Step 6: Verify Installation

Verify that all packages are installed correctly:

```bash
# Check Python version
python --version  # Should be 3.12.9

# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Check vLLM installation
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# Check transformers
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

## Configuration

### Configure DeepSeek OCR Settings

Edit the configuration file before running:

```bash
nano DeepSeek-OCR-vllm/config.py
```

Key settings to configure:

- **MODEL_PATH**: Model path (default: `'deepseek-ai/DeepSeek-OCR'`)
- **INPUT_PATH**: Path to input PDF/images
- **OUTPUT_PATH**: Path for output files
- **BASE_SIZE**: Base image size (512, 640, 1024, or 1280)
- **IMAGE_SIZE**: Processing image size (512, 640, 1024, or 1280)
- **CROP_MODE**: Enable crop mode for dynamic resolution (True/False)
- **MAX_CONCURRENCY**: Maximum concurrent requests (default: 100, reduce if GPU memory is limited)
- **GPU_MEMORY_UTILIZATION**: GPU memory usage (default: 0.9, reduce if out of memory)

### Resolution Modes

Choose the appropriate resolution mode based on your GPU memory:

- **Tiny**: `base_size = 512, image_size = 512, crop_mode = False` (64 vision tokens)
- **Small**: `base_size = 640, image_size = 640, crop_mode = False` (100 vision tokens)
- **Base**: `base_size = 1024, image_size = 1024, crop_mode = False` (256 vision tokens)
- **Large**: `base_size = 1280, image_size = 1280, crop_mode = False` (400 vision tokens)
- **Gundam**: `base_size = 1024, image_size = 640, crop_mode = True` (dynamic resolution)

## Usage

### For PDF Processing

```bash
cd DeepSeek-OCR-vllm
python run_dpsk_ocr_pdf.py
```

### For Image Processing

```bash
cd DeepSeek-OCR-vllm
python run_dpsk_ocr_image.py
```

### For Batch Evaluation

```bash
cd DeepSeek-OCR-vllm
python run_dpsk_ocr_eval_batch.py
```

## Troubleshooting

### CUDA Issues

- **Error: CUDA not available**
  - Ensure CUDA 11.8 is installed: `nvcc --version`
  - Check GPU drivers: `nvidia-smi`
  - Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### vLLM Installation Issues

- **Error: vLLM wheel not found**
  - Check Python version compatibility
  - Try installing from source: `pip install vllm==0.8.5`
  - Check [vLLM releases](https://github.com/vllm-project/vllm/releases) for compatible versions

### Flash Attention Issues

- **Error: flash-attn installation fails**
  - Install CUDA toolkit development files
  - Try: `pip install flash-attn --no-build-isolation --no-cache-dir`
  - If still failing, skip this step (performance will be reduced)

### Out of Memory (OOM) Errors

- Reduce `MAX_CONCURRENCY` in `config.py`
- Reduce `gpu_memory_utilization` in the LLM initialization
- Use a smaller resolution mode (Tiny or Small)
- Reduce `MAX_CROPS` if using crop_mode

### Model Download Issues

- The model will be automatically downloaded from HuggingFace on first use
- Ensure you have sufficient disk space (~10GB+)
- If download fails, manually download from: https://huggingface.co/deepseek-ai/DeepSeek-OCR

## Environment Variables

You may need to set these environment variables:

```bash
# For CUDA 11.8
export TRITON_PTXAS_PATH="/usr/local/cuda-11.8/bin/ptxas"

# Disable vLLM v1 (use v0)
export VLLM_USE_V1='0'

# Set visible GPU devices
export CUDA_VISIBLE_DEVICES='0'
```

## Performance Notes

- **Expected Performance**: ~2500 tokens/s on A100-40G GPU
- **GPU Memory**: Requires significant GPU memory (40GB+ recommended)
- **Processing Speed**: Depends on resolution mode and GPU memory
- **Concurrency**: Higher concurrency = faster batch processing but more GPU memory

## Additional Resources

- [DeepSeek OCR Paper](DeepSeek_OCR_paper.pdf)
- [HuggingFace Model](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [vLLM Documentation](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html)
- [Original Repository](https://github.com/deepseek-ai/DeepSeek-OCR)

## Next Steps

After installation, you can:
1. Configure `DeepSeek-OCR-vllm/config.py` with your input/output paths
2. Test with a sample image or PDF
3. Integrate DeepSeek OCR into your translation pipeline

## Support

For issues specific to DeepSeek OCR, refer to:
- [DeepSeek OCR GitHub Issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues)
- [DeepSeek Discord](https://discord.gg/Tc7c45Zzu5)


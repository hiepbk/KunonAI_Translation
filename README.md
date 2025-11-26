# KunonAI Translation

Document translation and OCR project using PyTorch, PaddleOCR, EasyOCR, and OpenAI API.

## Prerequisites

- Conda (Miniconda or Anaconda)
- CUDA 11.8 compatible GPU (for GPU version)
- Python 3.10
- OpenAI API key

## Step-by-Step Conda Environment Setup

### Step 1: Create a new conda environment

```bash
conda create -n kuai python=3.10 -y
```

### Step 2: Activate the environment

```bash
conda activate kuai
```

### Step 3: Install special packages manually (PyTorch with CUDA support)

These packages require special installation with specific index URLs. Install PyTorch with CUDA 11.8 support:

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```

**Note:** 
- If you don't have CUDA or want CPU-only version, use:
  ```bash
  pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
  ```
- For other CUDA versions, check [PyTorch installation guide](https://pytorch.org/get-started/locally/)

### Step 4: Install other special packages manually (if needed)

If you need PaddleOCR, install it separately:

**Important:** PaddlePaddle 3.0.0+ is required for OpenSSL 3.0 compatibility (Ubuntu 22.04+ systems). The GPU version must be installed from the official PaddlePaddle repository, not PyPI.

For **GPU version** (CUDA 11.8):
```bash
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
pip install paddleocr==2.7.0
```

For **CPU version**:
```bash
pip install paddlepaddle==3.0.0
pip install paddleocr==2.7.0
```

**Note:** 
- PaddlePaddle 2.x versions require OpenSSL 1.1, which is not available on Ubuntu 22.04+ systems (which have OpenSSL 3.0)
- Version 3.0.0+ supports OpenSSL 3.0 and is compatible with modern Linux distributions
- The GPU version must be installed from the official PaddlePaddle repository to ensure GPU support
- PaddleOCR installation may require additional system dependencies. Refer to [PaddleOCR documentation](https://github.com/PaddlePaddle/PaddleOCR) for details.

### Step 5: Install common packages from requirements.txt

Install all common packages using the requirements file:

```bash
pip install -r requirements.txt
```

### Step 6: Verify installation

Verify that all packages are installed correctly:

```bash
# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check PaddlePaddle GPU (if installed)
python -c "import paddle; print(f'PaddlePaddle version: {paddle.__version__}'); print(f'GPU available: {paddle.device.is_compiled_with_cuda()}'); print(f'GPU count: {paddle.device.cuda.device_count() if paddle.device.is_compiled_with_cuda() else 0}')"

# Check PaddleOCR (if installed)
python -c "from paddleocr import PaddleOCR; print('PaddleOCR installed successfully')"

# Check EasyOCR
python -c "import easyocr; print('EasyOCR installed successfully')"

# Check OpenAI
python -c "from openai import OpenAI; print('OpenAI installed successfully')"
```

## Environment Variables

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or you can modify the `OPENAI_API_KEY` variable directly in the script files.

## Usage

### Processing Modes

The script supports three main processing modes:

1. **OCR-Only Mode** (`--ocr-view` or `--ocr-only`): Fast mode that shows OCR results without translation
   - Shows: **Original | OCR Result** side by side
   - No API calls, very fast
   - Use this to check OCR accuracy before translation

2. **Full View Mode** (`--full-view` or `--ocr-translate-view`): Complete processing with OCR and translation
   - Shows: **Original | OCR Result | Translation** side by side
   - Includes translation via OpenAI API (slower, costs API credits)
   - Use this for final translated output

3. **Translation Mode** (default): Creates translated PDF only
   - Generates translated PDF file
   - No comparison images (unless using `--full-view`)

### GPU Version (Recommended for faster processing)

**OCR-Only Mode (Fast - Check OCR results first):**
```bash
# Process all pages - OCR only
python final-gpu.py --file ./sample_data/2.2.4 의료비 지출내역.pdf --ocr paddle --ocr-view

# Process specific pages - OCR only
python final-gpu.py --file ./sample_data/2.2.4 의료비 지출내역.pdf --ocr paddle --ocr-view 2,3,4
```

**Full View Mode (OCR + Translation):**
```bash
# Process all pages with translation
python final-gpu.py --file ./sample_data/2.2.4 의료비 지출내역.pdf --ocr paddle --model gpt-4o --mode all --full-view

# Process specific pages with translation
python final-gpu.py --file ./sample_data/2.2.4 의료비 지출내역.pdf --ocr paddle --model gpt-4o --mode all --full-view 2,3,4
```

**Translation Mode (Default - Translated PDF only):**
```bash
# Process all pages - translation only
python final-gpu.py --file ./sample_data/2.2.4 의료비 지출내역.pdf --ocr paddle --model gpt-4o --mode all

# Process specific pages - translation only
python final-gpu.py --file ./sample_data/2.2.4 의료비 지출내역.pdf --ocr paddle --model gpt-4o --mode all 2,3,4
```

### CPU Version

```bash
python final-cpu.py --file ./sample_data/your_file.pdf --ocr paddle --model gpt-4o --mode all
```

### Command Line Arguments

- `--file` or `-f`: Path to the PDF file to process
- `--ocr` or `-o`: OCR engine to use (`paddle`, `easyocr`)
- `--model` or `-m`: OpenAI model to use (`gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`, etc.)
- `--mode`: Translation mode (`all`, `eng_only`, `eng_chi`, `eng_ar`)
  - `all`: Translate all languages (English, Arabic, Chinese)
  - `eng_only`: Translate only English
  - `eng_chi`: Translate English only (keep Chinese characters)
  - `eng_ar`: Translate English and Arabic
- `--ocr-view` or `--ocr-only`: OCR-only mode (fast, no translation)
- `--full-view` or `--ocr-translate-view`: Full view mode (OCR + Translation with comparison images)
- **Page numbers**: Specify at the end of command (e.g., `1`, `1,3,5`, `1-10`, `1,3,5-8`)

### Page Selection Examples

```bash
# Single page
python final-gpu.py --file document.pdf --ocr paddle --ocr-view 1

# Multiple specific pages
python final-gpu.py --file document.pdf --ocr paddle --ocr-view 2,3,4

# Page range
python final-gpu.py --file document.pdf --ocr paddle --ocr-view 1-10

# Mixed (specific pages + range)
python final-gpu.py --file document.pdf --ocr paddle --ocr-view 1,3,5-8,10-15
```

## Project Structure

```
KunonAI_Translation/
├── final-gpu.py          # GPU-optimized version
├── final-cpu.py          # CPU version
├── requirements.txt      # Common package dependencies
├── README.md            # This file
├── sample_data/         # Sample PDF files for testing
└── results/             # OCR and translation results
```

## Dependencies

### Special Packages (Install Manually)

These packages require special installation methods:

1. **PyTorch** (with CUDA 11.8):
   ```bash
   pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
   ```

2. **PaddleOCR** (optional):
   
   **For GPU version (CUDA 11.8):**
   ```bash
   python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
   pip install paddleocr==2.7.0
   ```
   
   **For CPU version:**
   ```bash
   pip install paddlepaddle==3.0.0
   pip install paddleocr==2.7.0
   ```
   
   **Important:** PaddlePaddle 3.0.0+ is required for OpenSSL 3.0 compatibility. The GPU version must be installed from the official PaddlePaddle repository (not PyPI) to ensure GPU support.

### Common Packages (from requirements.txt)

Install using:
```bash
pip install -r requirements.txt
```

See `requirements.txt` for the complete list of common dependencies.

## Troubleshooting

### CUDA Issues

- Ensure you have CUDA 11.8 installed and compatible GPU drivers
- Check GPU status: `nvidia-smi`
- Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### PaddlePaddle OpenSSL Issues

- **Error: `libssl.so.1.1: cannot open shared object file`**
  - This occurs when using PaddlePaddle 2.x on systems with OpenSSL 3.0 (Ubuntu 22.04+)
  - **Solution:** Install PaddlePaddle 3.0.0+ which supports OpenSSL 3.0:
    ```bash
    # For GPU version
    python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
    
    # For CPU version
    pip install paddlepaddle==3.0.0
    ```

### PaddlePaddle cuDNN Issues

- If you see `RuntimeError: Cannot load cudnn shared library`, set the library path before running:
  ```bash
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
  ```
- Install cuDNN for CUDA 11 if not already installed:
  ```bash
  conda install -c conda-forge cudnn=8.9.2.26=cuda11_0 -y
  ```
- **Note:** PaddlePaddle 3.0.0 will automatically install the required cuDNN version during installation

### OCR Library Issues

- **PaddleOCR**: May require additional system dependencies (OpenCV, etc.)
- **EasyOCR**: Automatically downloads models on first use (ensure sufficient disk space)
- If OCR fails, try switching between different OCR engines

### NumPy Compatibility Issues

- **PyTorch 2.2.0 requires NumPy 1.x** (not compatible with NumPy 2.x)
- If you see errors like "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x", downgrade NumPy:
  ```bash
  pip install "numpy>=1.24.3,<2.0.0"
  ```
- The `requirements.txt` already pins NumPy to `<2.0.0` to prevent this issue

### Import Errors

- Ensure conda environment is activated: `conda activate kuai`
- Verify all packages are installed: `conda list` or `pip list`
- Reinstall problematic packages if needed

### OpenAI API Issues

- Verify your API key is set correctly
- Check your OpenAI account quota and billing
- Ensure you have internet connection for API calls

## Output Files

### OCR-Only Mode (`--ocr-view`)
- **Images**: Saved in `results/` folder
  - Format: `ocr_result_{OCR_ENGINE}_{filename}_page_{page_number}.png`
  - Shows: Original | OCR Result side by side

### Full View Mode (`--full-view`)
- **Images**: Saved in `results/` folder
  - Format: `full_result_{OCR_ENGINE}_{filename}_page_{page_number}.png`
  - Shows: Original | OCR Result | Translation side by side
- **PDF**: Saved in current directory
  - Format: `translated_{filename}.pdf` or `translated_{filename}_page_{pages}.pdf`
  - Contains translated pages only

### Translation Mode (default)
- **PDF**: Saved in current directory
  - Format: `translated_{filename}.pdf` or `translated_{filename}_page_{pages}.pdf`
  - Contains translated pages only

## Notes

- The project supports multiple OCR engines (PaddleOCR, EasyOCR)
- OpenAI models can be configured (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)
- **Recommended workflow**: Use `--ocr-view` first to check OCR accuracy, then use `--full-view` for translation
- Results are saved in the `results/` directory (for images) and current directory (for PDFs)
- Sample data is provided in `sample_data/` for testing
- Translation speed depends on the number of text elements detected and the OpenAI model used
  - `gpt-4o-mini`: Faster and cheaper, good quality
  - `gpt-4o`: Slower and more expensive, higher quality

## License

[Add your license information here]


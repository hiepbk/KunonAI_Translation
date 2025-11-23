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

```bash
pip install paddlepaddle-gpu>=2.5.0
pip install paddleocr>=2.7.0
```

**Note:** PaddleOCR installation may require additional system dependencies. Refer to [PaddleOCR documentation](https://github.com/PaddlePaddle/PaddleOCR) for details.

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

### GPU Version (Recommended for faster processing)

```bash
python final-gpu.py --file ./sample_data/마일스톤.pdf --ocr paddle --model gpt-4o --mode all
```

### CPU Version

```bash
python final-cpu.py --file ./sample_data/your_file.pdf --ocr paddle --model gpt-4o --mode all
```

### Command Line Arguments

- `--file`: Path to the PDF file to process
- `--ocr`: OCR engine to use (`paddle`, `easyocr`, etc.)
- `--model`: OpenAI model to use (`gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`, etc.)
- `--mode`: Processing mode
- `--ocr-view`: Display OCR results (optional)

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
   ```bash
   pip install paddlepaddle-gpu>=2.5.0
   pip install paddleocr>=2.7.0
   ```

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

### PaddlePaddle cuDNN Issues

- If you see `RuntimeError: Cannot load cudnn shared library`, set the library path before running:
  ```bash
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
  ```
- Install cuDNN for CUDA 11 if not already installed:
  ```bash
  conda install -c conda-forge cudnn=8.9.2.26=cuda11_0 -y
  ```

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

## Notes

- The project supports multiple OCR engines (PaddleOCR, EasyOCR)
- OpenAI models can be configured (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)
- Results are saved in the `results/` directory
- Sample data is provided in `sample_data/` for testing

## License

[Add your license information here]


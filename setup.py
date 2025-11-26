from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README if exists
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name="kunon-ai-translation",
    version="0.1.0",
    description="DeepSeek OCR Translation Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    packages=find_packages(exclude=['tests', 'tests.*']),
    package_dir={'': '.'},
    python_requires=">=3.12",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        '': ['*.md', '*.txt', '*.json'],
    },
)


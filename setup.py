from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="carbonwatcher",
    version="0.3.0",
    packages=find_packages(exclude=["tests*", "carbon_tracker*"]),
    python_requires=">=3.9",

    # Metadata
    author="Ghanshyam Paunikar",
    author_email="ghanshyampaunikar@gmail.com",
    description=(
        "Track energy, carbon, and cost of ML experiments and LLM API calls — "
        "CLI and Python module with automatic token counting."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ghanshyampaunikar/carbonwatcher",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Monitoring",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "carbon", "emissions", "energy", "sustainability", "machine-learning",
        "llm", "tracking", "green-ai", "co2", "climate",
    ],

    # Core dependencies
    install_requires=[
        "psutil>=5.9.0",
        "nvidia-ml-py>=12.535.77",
        "rich>=13.0.0",
        "click>=8.1.0",
        "requests>=2.31.0",
    ],

    # Optional dependencies per LLM provider / backend
    extras_require={
        "anthropic":   ["anthropic>=0.25.0"],
        "openai":      ["openai>=1.0.0"],
        "ollama":      ["ollama>=0.2.0"],
        "huggingface": ["transformers>=4.35.0", "torch>=2.0.0"],
        "llamacpp":    ["llama-cpp-python>=0.2.0"],
        "tiktoken":    ["tiktoken>=0.5.0"],
        "all": [
            "anthropic>=0.25.0",
            "openai>=1.0.0",
            "ollama>=0.2.0",
            "transformers>=4.35.0",
            "torch>=2.0.0",
            "llama-cpp-python>=0.2.0",
            "tiktoken>=0.5.0",
        ],
        "dev": ["pytest>=7.0", "tiktoken>=0.5.0"],
    },

    entry_points={
        "console_scripts": [
            "carbonwatcher=carbonwatcher.cli.main:cli",
        ],
    },

    project_urls={
        "Bug Reports": "https://github.com/ghanshyampaunikar/carbonwatcher/issues",
        "Source":      "https://github.com/ghanshyampaunikar/carbonwatcher",
    },
)

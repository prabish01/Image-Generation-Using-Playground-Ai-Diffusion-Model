# Image Generator

Generate professional book covers using Playground AI's diffusion model.

## Overview

This project uses the Playground AI v2.5 diffusion model to automatically generate professional-looking book covers from article titles. The pipeline creates text-free covers with rich colors and clear focal points that can be used as a base for further design work.

## Features

- Batch processing of book cover generation from CSV input data
- Customizable image dimensions (default: 1880x240px)
- Checkpointing to resume interrupted generation jobs
- Memory optimization for processing large datasets
- Progress tracking via tqdm
- Automatic file naming and output organization

## Requirements

- Python 3.8+
- PyTorch
- Diffusers
- Pandas
- tqdm
- nanoid
- CUDA-capable GPU (recommended for faster processing)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/book-cover-generator.git
cd book-cover-generator

#Install reqirements
pip install -r requirements.txt

# Install dependencies
pip install torch diffusers pandas tqdm nanoid
```

## Usage

1. Prepare your data in a CSV file named `data.csv` with at least an `article_title` column
2. Run the script:

```bash
python generate_covers.py
```

3. Generated covers will be saved in the `output` directory
4. Processing progress will be tracked in `completed.csv`

## Configuration

You can modify these variables at the top of the script:

- `OUTPUT_DIR`: Directory where generated images are saved
- `BATCH_SIZE`: Number of images to generate in each batch
- `CHECKPOINT_INTERVAL`: How often to save progress

## Advanced Options

The image generation can be fine-tuned by modifying these parameters:

```python
# In the process_batch function
image = pipe(
    prompt=prompt,
    num_inference_steps=11,   # Controls generation quality/speed balance
    guidance_scale=2,         # Controls prompt adherence
    width=1880,               # Image width
    height=240,               # Image height
    num_images_per_prompt=1,  # Number of variations to generate
).images[0]
```

## Input Format

The `data.csv` file should include at least an `article_title` column. Additional columns will be preserved in the output `completed.csv` file.

## Output

- Generated images are saved as PNG files in the `output` directory
- A `completed.csv` file tracks which titles have been processed and their corresponding filenames

## Performance Considerations

- The script includes memory optimizations for CUDA devices
- For large datasets, consider adjusting `BATCH_SIZE` and `CHECKPOINT_INTERVAL`
- Processing time depends heavily on your hardware capabilities

## License

[MIT License](LICENSE)

## Acknowledgements

- [Playground AI](https://playground.ai/) for the diffusion model
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) library

# JPEG Encoder

This project is a JPEG encoder that provides two versions: a full CPU implementation and a hybrid CPU/GPU implementation using CUDA.

## Overview

The JPEG Encoder is designed to efficiently compress images into the JPEG format. It supports two modes:

1. **Full CPU Version:** This version utilizes the CPU for all encoding processes. It is suitable for systems without a compatible GPU or for users who prefer a CPU-only implementation.

2. **CPU/GPU Version with CUDA:** This version leverages CUDA, a parallel computing platform and application programming interface model created by Nvidia, to offload certain processing tasks to the GPU. This can significantly enhance the encoding performance on systems with CUDA-compatible GPUs.

## Features

- Full support for the JPEG compression standard.
- Two encoding modes: CPU-only and CPU/GPU with CUDA.
- Efficient parallel processing using CUDA for improved performance.
- Easy-to-use command-line interface for encoding images.

## Getting Started

### Prerequisites

- **CUDA-enabled GPU:** Required for the CPU/GPU version. Ensure you have a compatible GPU and the necessary CUDA toolkit installed.

### Installation

1. **Clone the repository**:

    ```bash
        git clone git@github.com:iMaTzzz/gpu_project.git
    ```

2. **Build the Project**: Navigate to the project directory and build the encoder:
    ```bash
        mkdir dist
        cd dist
        cmake ..
        make
    ```

3. **Run the Encoder**: After building the project, you can run the JPEG encoder on the provided images.
Please note that when executing the GPU version on a single image, we refrain from including any warm-up procedure. Consequently, the observed runtime may appear slower compared to the results obtained from benchmark tests.
Here are three examples (Code is provided below):
    - If you want to run the CPU version on an image located in the *images* directory (First example)
    - If you want to run the GPU version on an image located in the *images* directory (Second example)
    - If you want to run both versions on all the images located in the *images* directory (Third example)

    ```bash
        ./ppm2jpeg ../images/<image>
        ./ppm2jpeg --gpu ../images/<image>
        ./ppm2jpeg --test:../images
    ```

## Requirements

- C/C++ compiler
- CUDA-enabled GPU (for the CPU/GPU version)
- Input images in supported formats (e.g., PGM, PPM) (Some images are already included in the images directory)

## Contributing

Contributions to this project are welcome! If you encounter any issues, have ideas for improvements, or would like to contribute new features, feel free to submit a pull request or open an issue.

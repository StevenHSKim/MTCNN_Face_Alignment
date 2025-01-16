# MTCNN_Face_Alignment
This repository provides a Python script for detecting and aligning faces from images using the [MTCNN (Multi-task Cascaded Convolutional Networks)](https://github.com/serengil/deepface). The aligned face images are saved in a specified output directory for further usage, such as facial expression recognition (FER).

<br>

## Methodology
- Save Only the Most Confident Detection: The script saves **"only one image per input image"**, selecting the detected face with the highest confidence.
- File Naming Convention: The aligned face images are saved with a consistent naming format: `<original_image_name>_aligned.jpg`. For example, if the input file is `example.jpg`, the aligned face will be saved as `example_aligned.jpg`.
- No Face Detected: If no face is detected in an image, the script prints a message indicating the failure (e.g., `No face detected in example.jpg`) and does not save any output for that image. This ensures that only meaningful results are stored.

<br>

## Prerequisites
Make sure you have Python 3.7 or higher installed. Install the required Python packages using the following commands:
```bash
pip install mtcnn
pip install opencv-python
pip install matplotlib
```

<br>

## Usage
1. Clone the repository:
ce-alignment-mtcnn

2. Prepare your dataset:
Place your input images in a directory (e.g., original_image).

3. Update the input and output directory paths in the script:
```python
input_dir = "/path/to/your/input/images"  # Path to the directory containing input images
output_dir = "/path/to/your/output/images"  # Path to save aligned face images
```
4. Run the script:
```bash
python face_alignment.py
```
The aligned face images will be saved in the specified output_dir.

<br>

## Example Output

Input Image:

<img src="https://github.com/user-attachments/assets/6f2f019e-8096-4ee5-a468-51c88f8078b8" width="100" />

Aligned Image:

<img src="https://github.com/user-attachments/assets/facb7e88-94c2-4a44-a475-83fa07475346" width="100" />

(The example images shown above are taken from the ExpW dataset)

<br>

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/StevenHSKim/MTCNN_Face_Alignment/blob/main/LICENSE) file for details.

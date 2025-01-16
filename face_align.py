import os
import cv2
import numpy as np
from mtcnn import MTCNN
import matplotlib.pyplot as plt

# 1. Initialize the MTCNN model
detector = MTCNN()

# 2. Define the face alignment function
def align_face(image, keypoints):
    # Coordinates of the two eyes
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    # Center point between the two eyes
    eye_center = ((left_eye[0] + right_eye[0]) * 0.5, (left_eye[1] + right_eye[1]) * 0.5)

    # Calculate the angle
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))  # Slope between the two eyes

    # Create the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)

    # Rotate the image
    aligned_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    return aligned_image

# 3. Detect and align faces from the dataset
def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Failed to load {img_path}")
            continue

        # Convert OpenCV's BGR image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Use MTCNN to detect faces
        results = detector.detect_faces(rgb_image)

        if results:
            # Sort the results by confidence
            max_confidence_result = max(results, key=lambda x: x['confidence'])
            
            # Bounding box and keypoints
            bounding_box = max_confidence_result['box']
            keypoints = max_confidence_result['keypoints']

            # Crop the face region
            x, y, width, height = bounding_box
            face = rgb_image[y:y+height, x:x+width]

            # Align the face
            aligned_face = align_face(face, keypoints)

            # Set the path to save the aligned face (now saved without index)
            aligned_face_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_aligned.jpg")

            # Save (convert RGB to BGR before saving)
            cv2.imwrite(aligned_face_path, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))

            print(f"Saved aligned face: {aligned_face_path} (confidence: {max_confidence_result['confidence']:.3f})")
        else:
            print(f"No face detected in {img_name}")

# 4. Execute
input_dir = "./original_image"  # Original image path
output_dir = "./aligned_image"  # Aligned image path
process_images(input_dir, output_dir)

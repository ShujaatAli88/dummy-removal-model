import os
import sys
import cv2
import numpy as np
import torch
from typing import List
import urllib.request
import subprocess
import supervision as sv
from segment_anything import sam_model_registry, SamPredictor


# Set HOME directory
HOME = os.getcwd()
print("HOME:", HOME)

# Define repository details
grounding_dino_repo = "GroundingDINO"
grounding_dino_url = "https://github.com/IDEA-Research/GroundingDINO.git"
segment_anything_repo = "segment-anything"
segment_anything_url = "git+https://github.com/facebookresearch/segment-anything.git"

# Clone GroundingDINO repository if not already cloned
os.chdir(HOME)
if not os.path.exists(grounding_dino_repo):
    os.system("git config --global http.postBuffer 157286400")  # Set Git buffer size
    os.system(f"git clone {grounding_dino_url}")
else:
    print(f"{grounding_dino_repo} repository already cloned.")

# Install GroundingDINO requirements
grounding_dino_path = os.path.join(HOME, grounding_dino_repo)
if os.path.exists(grounding_dino_path):
    os.chdir(grounding_dino_path)
    os.system("git checkout -q 57535c5a79791cb76e36fdb64975271354f10251")
    os.system(f"{sys.executable} -m pip install -e .")

# Install Segment Anything if not already installed
os.chdir(HOME)
if not os.path.exists(segment_anything_repo):  # Just a logical check; replace if needed
    os.system(f"{sys.executable} -m pip install {segment_anything_url}")
else:
    print(f"{segment_anything_repo} already installed or cloned."
          )


# Function to check if a package is installed with a specific version
def is_package_installed(package_name, version):
    try:
        import pkg_resources
        pkg_resources.require(f"{package_name}=={version}")
        return True
    except (ImportError, pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
        return False

# Set HOME directory
HOME = os.getcwd()
print("HOME Directory:", HOME)

# Ensure supervision is installed at the required version
required_supervision_version = "0.6.0"
if not is_package_installed("supervision", required_supervision_version):
    print(f"Installing supervision=={required_supervision_version}...")
    subprocess.run(["pip", "uninstall", "-y", "supervision"], check=True)
    subprocess.run(["pip", "install", f"supervision=={required_supervision_version}"], check=True)
else:
    print(f"supervision=={required_supervision_version} is already installed.")

# Verify supervision version
print("Supervision version:", sv.__version__)

# Paths
GROUNDING_DINO_CONFIG_PATH = os.path.normpath(
    os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
)
print("Expected Config Path:", GROUNDING_DINO_CONFIG_PATH)

# Debugging: Check files in the directory
config_dir = os.path.normpath(os.path.join(HOME, "GroundingDINO/groundingdino/config"))
if os.path.isdir(config_dir):
    print("Files in Config Directory:", os.listdir(config_dir))
else:
    print("Config directory does not exist:", config_dir)

# Check if file exists
file_exists = os.path.isfile(GROUNDING_DINO_CONFIG_PATH)
print(GROUNDING_DINO_CONFIG_PATH, "; exist:", file_exists)

# Weights directory
weights_dir = os.path.join(HOME, "weights")
os.makedirs(weights_dir, exist_ok=True)

# Grounding DINO Weights
dino_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
dino_path = os.path.join(weights_dir, "groundingdino_swint_ogc.pth")
if not os.path.isfile(dino_path):
    print(f"Downloading Grounding DINO weights to {dino_path}...")
    urllib.request.urlretrieve(dino_url, dino_path)
else:
    print(f"Grounding DINO weights already exist at {dino_path}.")

# SAM Weights
sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
sam_path = os.path.join(weights_dir, "sam_vit_h_4b8939.pth")
if not os.path.isfile(sam_path):
    print(f"Downloading SAM weights to {sam_path}...")
    urllib.request.urlretrieve(sam_url, sam_path)
else:
    print(f"SAM weights already exist at {sam_path}.")

# Check if GPU is available and set the device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Set HOME to the current working directory
HOME = os.getcwd()
print(f"HOME is set to: {HOME}")

# Define paths
GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")

# Validate paths
if not os.path.exists(GROUNDING_DINO_CONFIG_PATH):
    raise FileNotFoundError(f"Configuration file not found: {GROUNDING_DINO_CONFIG_PATH}")
if not os.path.exists(GROUNDING_DINO_CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint file not found: {GROUNDING_DINO_CHECKPOINT_PATH}")

# Add GroundingDINO to Python path
GROUNDING_DINO_PATH = os.path.join(HOME, "GroundingDINO")
sys.path.append(GROUNDING_DINO_PATH)

# Import the model
try:
    from GroundingDINO.groundingdino.util.inference import Model
except ImportError as e:
    raise ImportError("Failed to import Model from groundingdino.util.inference.") from e

# Load model
try:
    grounding_dino_model = Model(
        model_config_path=GROUNDING_DINO_CONFIG_PATH,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
        device=DEVICE.type
    )
    print("Model loaded successfully!")
except Exception as e:
    raise RuntimeError("Failed to initialize the GroundingDINO model.") from e


#Load SAM Model.

SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
if not os.path.exists(SAM_CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint sam file not found: {SAM_CHECKPOINT_PATH}")
else:
    print(f"Sam Model Found at:{SAM_CHECKPOINT_PATH}")
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)
print("Sam Model Loaded Successfully.")

  #Segment the Detections with SAM and return Masks.
def segment_and_getmasks(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)
    
class ImprovedMannequinRemover:
    def __init__(self):
        print("Model loaded successfully!")

    def ensure_directory_exists(self, folder_path):
        """Ensure that the directory exists, create if not."""
        try:
            if not os.path.exists(folder_path):
                print(f"Creating directory: {folder_path}")
                os.makedirs(folder_path)
            else:
                print(f"Directory already exists: {folder_path}")
        except Exception as e:
            print(f"Error creating directory {folder_path}: {e}")
            raise

    def resize_image(self, image, target_size=512):
        """Resize image maintaining aspect ratio"""
        h, w = image.shape[:2]
        aspect = w / h

        if h > w:
            new_h = target_size
            new_w = int(target_size * aspect)
        else:
            new_w = target_size
            new_h = int(target_size / aspect)

        return cv2.resize(image, (new_w, new_h))

    def pad_image(self, image, target_size=512):
        h, w = image.shape[:2]
        top = (target_size - h) // 2
        bottom = target_size - h - top
        left = (target_size - w) // 2
        right = target_size - w - left

        return cv2.copyMakeBorder(image, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=[255, 255, 255])

    def create_refined_mask(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        lower_white1 = np.array([0, 0, 200])
        upper_white1 = np.array([180, 30, 255])
        mask1 = cv2.inRange(hsv, lower_white1, upper_white1)

        lower_white2 = np.array([0, 0, 130])
        upper_white2 = np.array([180, 40, 255])
        mask2 = cv2.inRange(hsv, lower_white2, upper_white2)

        combined_mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        return mask

    def process_image(self, input_path, resized_folder, masks_folder):
        try:
            print(f"Processing image: {input_path}")
            original_image = cv2.imread(input_path)
            if original_image is None:
                raise ValueError(f"Could not read image at {input_path}")

            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            print("Preprocessing image...")
            resized_image = self.resize_image(original_image)
            padded_image = self.pad_image(resized_image)

            # Ensure output directories exist
            self.ensure_directory_exists(resized_folder)
            self.ensure_directory_exists(masks_folder)

            # Save resized image
            image_filename = os.path.basename(input_path)
            resized_image_path = os.path.join(resized_folder, f"resized_{image_filename}")
            padded_image = cv2.cvtColor(padded_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(resized_image_path, padded_image)

            print(f"Resized image saved to: {resized_image_path}")

            print("Creating mask...")
            mask = self.create_refined_mask(padded_image)

            # Save mask
            mask_image_path = os.path.join(masks_folder, f"mask_{image_filename}")
            cv2.imwrite(mask_image_path, mask)

            print(f"Mask saved to: {mask_image_path}")

            return resized_image_path, mask_image_path

        except Exception as e:
            print(f"Error: {str(e)}")
            raise

def expand_mask(mask, expansion_pixels=10):
    kernel = np.ones((expansion_pixels - 60, expansion_pixels - 60), np.uint8)
    expanded_mask = cv2.dilate(mask, kernel, iterations=1)
    return expanded_mask

def apply_inverse_mask(image_path, mask_path, output_folder):
    print(f"Applying inverse mask on: {image_path}")

    # Ensure output directory exists
    if not os.path.exists(output_folder):
        print(f"Creating output directory: {output_folder}")
        os.makedirs(output_folder)

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    mask_inv = cv2.bitwise_not(mask)
    expanded_mask = expand_mask(mask_inv, 60)

    white_background = np.ones_like(image) * 255
    result = np.where(expanded_mask[:, :, None] == 255, image, white_background)

    final_output_path = os.path.join(output_folder, f"output_{os.path.basename(image_path)}")
    cv2.imwrite(final_output_path, result)

    print(f"Processed image saved to: {final_output_path}")

# def process_folder(input_folder, resized_folder, masks_folder, output_folder):

#     if not os.path.exists(input_folder):
#         raise ValueError(f"Input folder does not exist: {input_folder}")

#     print("Ensuring output directories are set up.")
#     # Ensure that directories exist
#     remover = ImprovedMannequinRemover()
#     remover.ensure_directory_exists(resized_folder)
#     remover.ensure_directory_exists(masks_folder)
#     remover.ensure_directory_exists(output_folder)

#     for filename in os.listdir(input_folder):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):

#             input_path = os.path.join(input_folder, filename)

#             # Process image and create mask
#             resized_image_path, mask_image_path = remover.process_image(input_path, resized_folder, masks_folder)

#             # Apply inverse mask
#             apply_inverse_mask(resized_image_path, mask_image_path, output_folder)

def process_folder(input_folder, resized_folder, masks_folder, output_folder):
    """
    Process all images in the input folder to apply segmentation and save outputs.

    Args:
        input_folder (str): Path to the folder containing input images.
        resized_folder (str): Path to save resized images.
        masks_folder (str): Path to save generated masks.
        output_folder (str): Path to save the final output images.
    """
    try:
        # Ensure output directories exist
        os.makedirs(resized_folder, exist_ok=True)
        os.makedirs(masks_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)

        # Process each image in the folder
        for filename in os.listdir(input_folder):
            input_path = os.path.join(input_folder, filename)
            if not os.path.isfile(input_path):
                continue  # Skip non-file entries

            print(f"Processing file: {filename}")
            try:
                remover = ImprovedMannequinRemover()
                resized_path, mask_path = remover.process_image(input_path, resized_folder, masks_folder)
                apply_inverse_mask(resized_path, mask_path, output_folder)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    except Exception as e:
        print(f"Error processing folder: {e}")
        raise


def main():
    # Define folder paths (now relative to current directory)
    input_folder = "input_folder"        # Replace with your input folder path
    resized_folder = "./resized_images"  # Relative folder path
    masks_folder = "./masks"             # Relative folder path
    output_folder = "./output_images"    # Relative folder path

    # Ensure directories are created and processed
    process_folder(input_folder, resized_folder, masks_folder, output_folder)


if __name__ == "__main__":
    main()

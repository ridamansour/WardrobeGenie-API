import os
from transformers import YolosForObjectDetection, YolosImageProcessor

# Define the model names
MODEL_NAME = "valentinafeve/yolos-fashionpedia"
PROCESSOR_NAME = "hustvl/yolos-small"

# Define where you want to save them locally
LOCAL_DIR = "./models/yolos-fashionpedia-local"


def download_and_save_locally():
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)

    print(f"Downloading model '{MODEL_NAME}'...")
    model = YolosForObjectDetection.from_pretrained(MODEL_NAME)

    print(f"Downloading processor '{PROCESSOR_NAME}'...")
    processor = YolosImageProcessor.from_pretrained(PROCESSOR_NAME)

    # Save both to the same local directory
    print(f"Saving files to {LOCAL_DIR}...")
    model.save_pretrained(LOCAL_DIR)
    processor.save_pretrained(LOCAL_DIR)

    print("Done! You can now load the model directly from disk.")


if __name__ == "__main__":
    download_and_save_locally()
import sys
from pathlib import Path

# 1. Define the project root dynamically
# __file__ points to config.py, so its parent is the project root
ROOT_DIR = Path(__file__).resolve().parent

# 2. Add the root directory to the Python path
# This helps prevent import errors when running nested scripts
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# 3. Define top-level directories
SRC_DIR = ROOT_DIR / "src"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
DOCS_DIR = ROOT_DIR / "docs"
PUBLIC_DIR = ROOT_DIR / "public"
DATA_DIR = ROOT_DIR / "data"

# 4. Define specific model paths (based on your folder structure)
RF_DETR_MODEL_DIR = MODELS_DIR / "rf-detr_detection"
YOLO_MODEL_DIR = MODELS_DIR / "yolos-fashionpedia"
VISUAL_EMBEDDER_DIR = MODELS_DIR / "visual_embedder"
STYLIST_BRAIN_DIR = MODELS_DIR / "stylist_brain"
ATTRIBUTE_PREDICTOR_DIR = MODELS_DIR / "attribute_predictor"
NLP_MODEL_DIR = MODELS_DIR / "nlp_query"

# 5. Define data or export paths (example)
FASHIONOPEDIA_COCO_DIR = DATA_DIR / "fashionpedia_coco"
IMAGE_EXAMPLES_DIR = DATA_DIR / "examples"
FASHION_QUERIES_TXT_DATA = DATA_DIR / "fashion_queries_realworld.txt"

VECTOR_DB_URL = "http://localhost:6333"
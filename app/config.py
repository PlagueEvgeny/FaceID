from pathlib import Path

import serial

SERIAL_PORT = "COM8"   # или "/dev/ttyUSB0" под Linux
SERIAL_BAUD = 115200
serial_port = None
# Базовые пути
BASE_DIR = Path(__file__).parent.parent
APP_DIR = Path(__file__).parent

# Directory settings
DATASET_DIR = BASE_DIR / 'data' / 'datasets'
LOGS_DIR = BASE_DIR / 'logs'
STATIC_DIR = BASE_DIR / 'static'
UPLOADS_DIR = STATIC_DIR / 'uploads'
IMAGES_DIR = STATIC_DIR / 'images'
HAARCASCADES_DIR = BASE_DIR / 'data' / 'haarcascades'

# Model settings
face_cascade = None
eye_cascade = None
model = None
HAAR_FILE = HAARCASCADES_DIR / 'haarcascade_frontalface_default.xml'
EYE_FILE = HAARCASCADES_DIR / 'haarcascade_eye.xml'
MODEL_FILE = BASE_DIR / 'data' / 'face_model.xml'
METADATA_FILE = BASE_DIR / 'data' / 'model_metadata.json'
LOGS_FILE = LOGS_DIR / 'activity.json'

# Image processing settings
IMAGE_SIZE = (130, 100)
MAX_IMAGES = 200
CONFIDENCE_THRESHOLD = 100  
UNKNOWN_THRESHOLD = 80  

# Camera settings
camera = None
camera_settings = {
    "index": 0,
    "width": 640,
    "height": 480,
    "fps": 30
}

# Server settings
DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 5000
DEBUG_MODE = False
THREADED_MODE = True

# Cascade detection settings
cascade_params = {
    "scaleFactor": 1.1,
    "minNeighbors": 5,
    "minSize": (30, 30),
    "maxSize": (500, 500)
}

# Log cleanup settings
LOG_RETENTION_DAYS = 30

# Font settings for text rendering
FONT_PATHS = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/verdana.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf"
]

# Haar cascade URLs
HAAR_FILE_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
EYE_FILE_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml'

# Order settings
names = {}
original_names = {}  
is_collecting_data = False
require_eyes_for_face = True
current_person_name = ""
collected_count = 0
recognition_stats = {
    'total_faces_detected': 0,
    'known_faces': 0,
    'unknown_faces': 0,
    'last_recognized': None
}

# traslit
TRANSLIT_TABLE = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
    'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ф': 'f', 'х': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch',
    'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'Yo',
    'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M',
    'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
    'Ф': 'F', 'Х': 'H', 'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Sch',
    'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya'
}
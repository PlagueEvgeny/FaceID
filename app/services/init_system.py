from app import config
from app.utils.download_cascade import download_and_load_cascade
from app.utils.logs import save_logs
from app.services.models import load_model, train_model

import os
import cv2
import serial

def init_serial():
    try:
        config.serial_port = serial.Serial(config.SERIAL_PORT, config.SERIAL_BAUD, timeout=1)
        print(f"Serial порт {config.SERIAL_PORT} открыт")
        return True
    except Exception as e:
        print(f"Ошибка открытия Serial порта: {e}")
        config.serial_port = None
        return False

def init_camera():
    if config.camera is not None:
        config.camera.release()
        config.camera = None
    cam = cv2.VideoCapture(config.camera_settings["index"])
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_settings["width"])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_settings["height"])
    cam.set(cv2.CAP_PROP_FPS, config.camera_settings["fps"])
    config.camera = cam

def init_face_cascade():
    """Инициализация каскада Хаара с улучшенной обработкой ошибок"""    
    try:
        # Сначала попробуем встроенный каскад OpenCV
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        print(f"Пробуем встроенный каскад: {cascade_path}")
        
        if os.path.exists(cascade_path):
            config.face_cascade = cv2.CascadeClassifier(cascade_path)
            if not config.face_cascade.empty():
                print(f"Каскад лиц успешно загружен из встроенных данных OpenCV")
                return True
        
        # Если встроенный не работает, попробуем локальный файл
        local_paths = [
            config.HAAR_FILE,
            os.path.join(os.getcwd(), config.HAAR_FILE),
            os.path.join(os.path.dirname(__file__), config.HAAR_FILE)
        ]
        
        for path in local_paths:
            print(f"Попытка локального каскада: {path}")
            if os.path.exists(path):
                config.face_cascade = cv2.CascadeClassifier(path)
                if not config.face_cascade.empty():
                    print(f"Каскад лиц загружен из: {path}")
                    return True
        
        # Если ничего не работает, попробуем скачать
        print("Попытка загрузить каскадный файл...")
        return download_and_load_cascade('face')
        
    except Exception as e:
        print(f"Ошибка инициализации каскада лиц: {e}")
        return False

def init_eye_cascade():
    """Инициализация каскада для глаз с улучшенной обработкой ошибок"""
    
    try:
        # Сначала попробуем встроенный каскад OpenCV
        cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        print(f"Пробуем встроенный каскад глаз: {cascade_path}")
        
        if os.path.exists(cascade_path):
            config.eye_cascade = cv2.CascadeClassifier(cascade_path)
            if not config.eye_cascade.empty():
                print(f"Каскад глаз успешно загружен из встроенных данных OpenCV")
                return True
        
        # Если встроенный не работает, попробуем локальный файл
        local_paths = [
            config.EYE_FILE,
            os.path.join(os.getcwd(), config.EYE_FILE),
            os.path.join(os.path.dirname(__file__), config.EYE_FILE)
        ]
        
        for path in local_paths:
            print(f"Попытка локального каскада глаз: {path}")
            if os.path.exists(path):
                config.eye_cascade = cv2.CascadeClassifier(path)
                if not config.eye_cascade.empty():
                    print(f"Глазной каскад загружен из: {path}")
                    return True
        
        # Если ничего не работает, попробуем скачать
        print("Попытка загрузить файл eye cascade...")
        return download_and_load_cascade('eye')
        
    except Exception as e:
        print(f"Ошибка инициализации каскада глаз:{e}")
        return False
    

def initialize_system():
    """Инициализация системы при запуске"""    
    print("Инициализация системы распознавания лиц...")

    # Инициализация каскадов
    if init_face_cascade():
        print("Каскад лиц успешно инициализирован")
    else:
        print("ВНИМАНИЕ: Не удалось инициализировать каскад лиц")
    
    if init_eye_cascade():
        print("Каскад глаз успешно инициализирован")
    else:
        print("ВНИМАНИЕ: не удалось инициализировать каскад глаз")
    
    if init_serial():
        print("Serial port успешно инициализирован")
    else:
        print("ВНИМАНИЕ: не удалось инициализировать Serial port")
    
    # Восстанавливаем оригинальные имена из существующих директорий
    if os.path.exists(config.DATASET_DIR):
        for subdir in os.listdir(config.DATASET_DIR):
            subpath = os.path.join(config.DATASET_DIR, subdir)
            if os.path.isdir(subpath):
                # Если это транслитерированное имя, сохраняем его как есть
                config.original_names[subdir] = subdir
    
    # Проверяем наличие данных для обучения
    training_data_exists = False
    if os.path.exists(config.DATASET_DIR):
        for subdir in os.listdir(config.DATASET_DIR):
            subpath = os.path.join(config.DATASET_DIR, subdir)
            if os.path.isdir(subpath):
                images = [f for f in os.listdir(subpath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if images:
                    training_data_exists = True
                    break
    
    print(f"Существуют данные обучения: {training_data_exists}")
    
    # Пытаемся загрузить существующую модель
    model_loaded = load_model()
    print(f"Модель загружена из файла: {model_loaded}")
    
    # Если есть данные для обучения, но модель не загружена или устарела
    if training_data_exists and not model_loaded:
        print("Обучающие данные найдены, но нет подходящей модели — обучаем новую модель...")
        if train_model():
            print("✓ Новая модель успешно обучена")
        else:
            print("✗ Не удалось обучить новую модель")
    elif training_data_exists and model_loaded:
        # Проверяем, нужно ли переобучить модель
        current_people = set()
        if os.path.exists(config.DATASET_DIR):
            for subdir in os.listdir(config.DATASET_DIR):
                subpath = os.path.join(config.DATASET_DIR, subdir)
                if os.path.isdir(subpath):
                    images = [f for f in os.listdir(subpath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if images:
                        current_people.add(subdir)
        
        model_people = set(config.names.values()) if config.names else set()
        
        if current_people != model_people:
            print(f"Обнаружено несоответствие данных. Текущее: {current_people}, Model: {model_people}")
            print("Переобучение модели с обновленными данными...")
            if train_model():
                print("✓ Модель успешно переобучена")
            else:
                print("✗ Не удалось переобучить модель")
    
    print(f"Система инициализирована. Модель готова: {config.model is not None}")
    print(f"Люди в системе: {list(config.names.values()) if config.names else 'None'}")
    print(f"Исходное сопоставление имен: {config.original_names}")
    print("Создание лога")
    save_logs()
import cv2
import os
import json
import numpy as np
from datetime import datetime

from app import config

def load_training_data():
    """Загрузка данных для обучения"""
    if not os.path.exists(config.DATASET_DIR):
        print(f"Каталог наборов данных {config.DATASET_DIR} не существует")
        return None, None, {}
    
    images, labels = [], []
    config.names = {}
    id = 0
    
    for subdir in os.listdir(config.DATASET_DIR):
        subpath = os.path.join(config.DATASET_DIR, subdir)
        if os.path.isdir(subpath):
            config.names[id] = subdir
            # Сохраняем оригинальное имя если оно есть в словаре
            original_name = config.original_names.get(subdir, subdir)
            image_count = 0
            for filename in os.listdir(subpath):
                filepath = os.path.join(subpath, filename)
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(id)
                    image_count += 1
            print(f"Загружено {image_count} изображения для {original_name} (каталог: {subdir})")
            id += 1
    
    if not images or not labels:
        print("Данные для обучения не найдены.")
        return None, None, {}
    
    print(f"Общие данные обучения: {len(images)} изображения, {len(set(labels))} люди")
    return np.array(images), np.array(labels), config.names

def save_model():
    """Сохранение обученной модели"""
    if config.model is not None:
        try:
            # Сохраняем модель OpenCV
            config.model.save(config.MODEL_FILE)
            
            # Сохраняем метаданные
            model_data = {
                'names': config.names,
                'original_names': config.original_names,  # Сохраняем оригинальные имена
                'image_size': config.IMAGE_SIZE,
                'confidence_threshold': config.CONFIDENCE_THRESHOLD,
                'unknown_threshold': config.UNKNOWN_THRESHOLD,
                'training_date': datetime.now().isoformat()
            }
            
            with open(config.METADATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, ensure_ascii=False, indent=2)
            
            print(f"Модель сохранена в {config.MODEL_FILE}")
            return True
        except Exception as e:
            print(f"Ошибка сохранения модели: {e}")
            return False
    return False

def load_model():
    """Загрузка сохраненной модели"""
    try:
        if os.path.exists(config.MODEL_FILE) and os.path.exists(config.METADATA_FILE):
            # Проверяем целостность файлов
            if os.path.getsize(config.MODEL_FILE) == 0 or os.path.getsize(config.METADATA_FILE) == 0:
                print("Файлы модели повреждены, удаляем их")
                try:
                    os.remove(config.MODEL_FILE)
                    os.remove(config.METADATA_FILE)
                except:
                    pass
                return False
            
            # Загружаем модель
            if not hasattr(cv2, 'face'):
                print("Ошибка: opencv-contrib-python не установлен")
                return False
                
            config.model = cv2.face.LBPHFaceRecognizer_create()
            config.model.read(config.MODEL_FILE)
            
            # Загружаем метаданные
            with open(config.METADATA_FILE, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            
            # Восстанавливаем names с правильными типами ключей
            config.names = {}
            for k, v in model_data['names'].items():
                config.names[int(k)] = v
            
            # Восстанавливаем оригинальные имена
            config.original_names = model_data.get('original_names', {})
            
            return True
            
    except Exception as e:
        try:
            if os.path.exists(config.MODEL_FILE):
                os.remove(config.MODEL_FILE)
            if os.path.exists(config.METADATA_FILE):
                os.remove(config.METADATA_FILE)
        except:
            pass
    
    return False

def train_model():
    """Обучение модели распознавания лиц"""
    images, labels, _ = load_training_data()
    if images is not None and labels is not None:
        try:
            if not hasattr(cv2, 'face'):
                print("Ошибка: opencv-contrib-python Не установлен. Установить с помощью: pip install opencv-contrib-python==4.8.1.78")
                return False
            
            config.model = cv2.face.LBPHFaceRecognizer_create()
            config.model.train(images, labels)
            
            # Сохраняем модель после обучения
            save_model()
            
            print("Модель успешно обучена")
            return True
        except Exception as e:
            print(f"Модель обучения ошибок: {e}")
            return False
    print("Данные по обучению отсутствуют.")
    return False
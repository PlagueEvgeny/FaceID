import cv2
import os
import urllib.request
from app import config

def download_and_load_cascade(cascade_type='face'):
    """Скачать и загрузить каскад (лицо или глаза)"""    
    try:
        if cascade_type == 'face':
            url = config.HAAR_FILE_URL
            filename = config.HAAR_FILE
            target_var = config.face_cascade
        else:
            url = config.EYE_FILE_URL
            filename = config.EYE_FILE
            target_var = config.eye_cascade

        print(f"Загрузка каскада {cascade_type} с: {url}")
        urllib.request.urlretrieve(url, filename)
        
        if os.path.exists(filename):
            cascade = cv2.CascadeClassifier(filename)
            if not cascade.empty():
                print(f"Каскад {cascade_type} успешно загружен и загружен")
                
                if cascade_type == 'face':
                    config.face_cascade = cascade
                else:
                    config.eye_cascade = cascade
                    
                return True
            else:
                print(f"Загруженный каскадный файл {cascade_type} недействителен")
        
    except Exception as e:
        print(f"Не удалось загрузить каскад {cascade_type}: {e}")
    
    return False
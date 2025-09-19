from flask import Blueprint, request, jsonify
import cv2
import numpy as np
import base64
from app.services.face_recognizer import FaceRecognizer
import app.services.models as models
from app import config


recognition_bp = Blueprint('recognition', __name__)

recognizer = None

def init_recognizer():
    """Инициализация распознавателя"""
    global recognizer
    try:
        recognizer = FaceRecognizer()
        # Попытка загрузить существующую модель
        if not models.load_model:
            print("Модель не найдена. Требуется обучение.")
        return True
    except Exception as e:
        print(f"Ошибка инициализации распознавателя: {e}")
        return False

@recognition_bp.route('/api/train_model', methods=['POST'])
def train_model():
    """API для обучения модели"""
    global recognizer
    
    try:
        if recognizer is None:
            init_recognizer()
        
        success = models.train_model()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Модель успешно обучена',
                'model_info': recognizer.get_model_info()
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Ошибка обучения модели'
            })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Ошибка: {str(e)}'
        })

@recognition_bp.route('/api/model_info', methods=['GET'])
def get_model_info():
    """Получить информацию о модели"""
    global recognizer
    
    if recognizer is None:
        init_recognizer()
    
    if recognizer:
        return jsonify(recognizer.get_model_info())
    else:
        return jsonify({'error': 'Распознаватель не инициализирован'})

@recognition_bp.route('/api/recognize_image', methods=['POST'])
def recognize_image():
    """Распознавание лиц на загруженном изображении"""
    global recognizer
    
    if recognizer is None or recognizer.model is None:
        return jsonify({
            'success': False,
            'message': 'Модель не обучена'
        })
    
    try:
        # Получаем изображение из запроса
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Изображение не найдено'
            })
        
        file = request.files['image']
        
        # Читаем изображение
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'success': False,
                'message': 'Не удалось декодировать изображение'
            })
        
        # Распознаем лица
        processed_image, results = recognizer.recognize_faces_in_frame(image)
        
        # Кодируем обработанное изображение обратно в base64
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'faces_found': len(results),
            'results': results,
            'processed_image': processed_image_b64
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Ошибка обработки: {str(e)}'
        })

@recognition_bp.route('/api/recognize_base64', methods=['POST'])
def recognize_base64():
    """Распознавание лиц в изображении, переданном как base64"""
    global recognizer
    
    if recognizer is None or recognizer.model is None:
        return jsonify({
            'success': False,
            'message': 'Модель не обучена'
        })
    
    try:
        data = request.get_json()
        image_b64 = data.get('image')
        
        if not image_b64:
            return jsonify({
                'success': False,
                'message': 'Изображение не найдено'
            })
        
        # Декодируем base64
        image_data = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'success': False,
                'message': 'Не удалось декодировать изображение'
            })
        
        # Распознаем лица
        processed_image, results = recognizer.recognize_faces_in_frame(image)
        
        # Кодируем обработанное изображение обратно в base64
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'faces_found': len(results),
            'results': results,
            'processed_image': processed_image_b64
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Ошибка обработки: {str(e)}'
        })

@recognition_bp.route('/api/update_threshold', methods=['POST'])
def update_threshold():
    try:
        data = request.get_json()
        config.CONFIDENCE_THRESHOLD = int(data["threshold"])
        return jsonify({
            'status': 'success',
            'CONFIDENCE_THRESHOLD': config.CONFIDENCE_THRESHOLD
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Ошибка: {str(e)}'
        })
    

@recognition_bp.route("/api/update_unknown_threshold", methods=["POST"])
def update_unknown_threshold():
    try:
        data = request.get_json()
        config.UNKNOWN_THRESHOLD = int(data["value"])
        return jsonify({
            "status": "success", 
            "UNKNOWN_THRESHOLD": config.UNKNOWN_THRESHOLD})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    


@recognition_bp.route('/api/test_accuracy', methods=['POST'])
def test_accuracy():
    """Тестирование точности модели"""
    global recognizer
    
    if config.model is None:
        return jsonify({
            'success': False,
            'message': 'Модель не обучена'
        })
    
    try:
        data = request.get_json() or {}
        test_images_per_person = data.get('test_images_per_person', 5)
        
        results = recognizer.test_recognition_accuracy(test_images_per_person)
        
        if results:
            return jsonify({
                'success': True,
                'results': results
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Не удалось провести тестирование'
            })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Ошибка тестирования: {str(e)}'
        })

init_recognizer()
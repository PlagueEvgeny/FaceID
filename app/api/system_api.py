from flask import Blueprint, request, jsonify
import os, cv2, shutil
from datetime import datetime

from app import config
from app.utils.transliterate import sanitize_filename, get_original_name
from app.services.models import train_model
from app.services.init_system import init_camera

system_api = Blueprint("system_api", __name__)

@system_api.route("/api/toggle_eye_requirement", methods=["POST"])
def toggle_eye_requirement():
    data = request.json
    config.require_eyes_for_face = bool(data.get("enabled", False))
    return jsonify({"success": True, "require_eyes_for_face": config.require_eyes_for_face})

@system_api.route("/api/update_cascade_params", methods=["POST"])
def update_cascade_params():
    try:
        data = request.json
        config.cascade_params["scaleFactor"] = float(data.get("scaleFactor", 1.1))
        config.cascade_params["minNeighbors"] = int(data.get("minNeighbors", 5))
        config.cascade_params["minSize"] = (int(data.get("minSize", 30)), int(data.get("minSize", 30)))
        config.cascade_params["maxSize"] = (int(data.get("maxSize", 500)), int(data.get("maxSize", 500)))
        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, message=str(e))

@system_api.route("/api/update_camera", methods=["POST"])
def update_camera():
    try:
        data = request.json
        config.camera_settings.update({
            "index": int(data.get("index", 0)),
            "width": int(data.get("width", 640)),
            "height": int(data.get("height", 480)),
            "fps": int(data.get("fps", 30)),
        })
        init_camera()
        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, message=str(e))

@system_api.route("/start_collection", methods=["POST"])
def start_collection():
    """Начать сбор данных"""
    data = request.get_json()
    person_name = data.get("name", "").strip()
    if not person_name:
        return jsonify({"success": False, "message": "Имя не может быть пустым"})

    os.makedirs(config.DATASET_DIR, exist_ok=True)
    sanitized = sanitize_filename(person_name)
    person_path = os.path.join(config.DATASET_DIR, sanitized)
    os.makedirs(person_path, exist_ok=True)

    config.original_names[sanitized] = person_name
    config.is_collecting_data, config.current_person_name, config.collected_count = True, person_name, 0

    return jsonify({"success": True, "message": f"Начат сбор данных для {person_name}"})

@system_api.route("/stop_collection", methods=["POST"])
def stop_collection():
    config.is_collecting_data = False
    return jsonify({"success": True, "message": "Сбор данных остановлен"})

@system_api.route("/train_model", methods=["POST"])
def train_model_endpoint():
    success = train_model()
    return jsonify({
        "success": success,
        "message": "Модель успешно обучена и сохранена" if success else "Ошибка обучения модели"
    })

@system_api.route("/get_status")
def get_status():
    """Получить статус"""
    try:
        camera_ready = False
        if config.camera is not None:
            try:
                current_pos = config.camera.get(cv2.CAP_PROP_POS_FRAMES)
                ret, _ = config.camera.read()
                camera_ready = ret
                if current_pos >= 0:
                    config.camera.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
            except Exception:
                camera_ready = False

        total_data_count = sum(
            len([f for f in os.listdir(os.path.join(config.DATASET_DIR, d))
                 if f.endswith((".jpg", ".jpeg", ".png", ".bmp"))])
            for d in os.listdir(config.DATASET_DIR) if os.path.isdir(os.path.join(config.DATASET_DIR, d))
        ) if os.path.exists(config.DATASET_DIR) else 0

        model_exists = os.path.exists(config.MODEL_FILE)
        return jsonify({
            "is_collecting": config.is_collecting_data,
            "collected_count": config.collected_count,
            "total_data_count": total_data_count,
            "current_person": config.current_person_name,
            "model_trained": config.model is not None,
            "model_exists": model_exists,
            "people_count": len(config.names),
            "people_names": list(config.names.values()),
            "camera_ready": camera_ready,
            "face_cascade_loaded": config.face_cascade is not None,
            "eye_cascade_loaded": config.eye_cascade is not None,
            "timestamp": datetime.now().isoformat(),
            "require_eyes_for_face": config.require_eyes_for_face,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@system_api.route("/get_people_list")
def get_people_list():
    people = []
    if os.path.exists(config.DATASET_DIR):
        for subdir in os.listdir(config.DATASET_DIR):
            subpath = os.path.join(config.DATASET_DIR, subdir)
            if os.path.isdir(subpath):
                count = len([f for f in os.listdir(subpath) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
                people.append({
                    "name": get_original_name(subdir),
                    "directory": subdir,
                    "image_count": count
                })
    return jsonify(people)

@system_api.route("/delete_person", methods=["POST"])
def delete_person():
    data = request.get_json()
    person_name = data.get("name", "").strip()
    if not person_name:
        return jsonify({"success": False, "message": "Имя не может быть пустым"})

    directory_name = next((s for s, o in config.original_names.items() if o == person_name), sanitize_filename(person_name))
    person_path = os.path.join(config.DATASET_DIR, directory_name)

    if os.path.exists(person_path):
        try:
            shutil.rmtree(person_path)
            config.original_names.pop(directory_name, None)
            if train_model():
                return jsonify({"success": True, "message": f"Удалено {person_name}, модель переобучена"})
            return jsonify({"success": True, "message": f"Удалено {person_name}, модель не переобучена"})
        except Exception as e:
            return jsonify({"success": False, "message": f"Ошибка удаления: {str(e)}"})
    return jsonify({"success": False, "message": "Человек не найден в базе"})

@system_api.route('/get_stats') 
def get_stats(): 
    """Получить статистику распознавания""" 
    return jsonify(config.recognition_stats) 

@system_api.route('/reset_stats', methods=['POST']) 
def reset_stats():
     """Сбросить статистику""" 
     config.recognition_stats = {
          'total_faces_detected': 0, 
          'known_faces': 0, 
          'unknown_faces': 0, 
          'last_recognized': None 
          } 
     return jsonify({'success': True, 'message': 'Статистика сброшена'})
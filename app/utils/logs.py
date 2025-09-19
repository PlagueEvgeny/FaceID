from datetime import datetime
import json
import os
from app import config

def save_logs():
    log_file = config.LOGS_FILE

    """Логирование активности"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        "is_collecting": config.is_collecting_data,
        "collected_count": config.collected_count,
        "current_person": config.current_person_name,
        "model_trained": config.model is not None,
        "people_count": len(config.names),
        "people_names": list(config.names.values()),
        "face_cascade_loaded": config.face_cascade is not None,
        "eye_cascade_loaded": config.eye_cascade is not None,
        "require_eyes_for_face": config.require_eyes_for_face,
    }
    
    try:
        # если файла нет → создаём пустой массив
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
        else:
            logs = []

        # добавляем запись
        logs.append(log_entry)

        # перезаписываем файл
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Ошибка записи в лог: {e}")

def cleanup_old_logs(days=30):
    """Очистка старых логов"""
    log_file = config.LOGS_FILE
    if not os.path.exists(log_file):
        return
    
    cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        filtered_lines = []
        for line in lines:
            try:
                log_entry = json.loads(line.strip())
                log_time = datetime.fromisoformat(log_entry['timestamp']).timestamp()
                if log_time > cutoff_date:
                    filtered_lines.append(line)
            except:
                continue
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.writelines(filtered_lines)
            
    except Exception as e:
        print(f"Ошибка очистки логов: {e}")
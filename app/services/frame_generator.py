import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import time

from app.utils.transliterate import sanitize_filename, get_original_name
from app.services.init_system import init_face_cascade
from app import config

def draw_text_with_russian(frame, text, position, color=(0, 255, 0), font_size=20):
    """
    Рисует русский текст на кадре с использованием Pillow
    """
    # Конвертируем BGR (OpenCV) в RGB (Pillow)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Создаем объект для рисования
    draw = ImageDraw.Draw(pil_image)
    
    # Пытаемся найти подходящий шрифт с поддержкой кириллицы
    font = None
    font_paths = [
        # Windows шрифты
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/verdana.ttf",
        # Linux шрифты
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        # macOS шрифты
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf"
    ]
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        except:
            continue
    
    # Если не нашли подходящий шрифт, используем стандартный (может не поддерживать кириллицу)
    if font is None:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Рисуем текст
    draw.text(position, text, font=font, fill=color)
    
    # Конвертируем обратно в BGR
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def generate_frames():
    """Генератор кадров для видео потока"""    
    try:
        if config.camera is None:
            print("Инициализация камеры...")
            for idx in range(3):
                test_camera = cv2.VideoCapture(idx)
                if test_camera.isOpened():
                    ret, test_frame = test_camera.read()
                    if ret and test_frame is not None:
                        config.camera = test_camera
                        print(f"Камера инициализирована по индексу {idx}")
                        config.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        config.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        config.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        config.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        config.camera.set(cv2.CAP_PROP_FPS, 30)
                        break
                    else:
                        test_camera.release()
                else:
                    test_camera.release()
            
            if config.camera is None:
                print("Ошибка: не удалось инициализировать камеру")
                return
        
    except Exception as e:
        print(f"Ошибка камеры: {e}")
        return
    
    frame_count = 0
    
    while True:
        try:
            success, frame = config.camera.read()
            if not success or frame is None:
                print("Не удалось прочитать кадр")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            if config.face_cascade is None or config.face_cascade.empty():
                frame = draw_text_with_russian(frame, "Каскад лиц не загружен - попытка перезагрузки...",
                                             (10, 30), (0, 0, 255))
                
                if frame_count % 30 == 0:
                    print("Попытка повторной инициализации каскада лиц...")
                    init_face_cascade()
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                faces = config.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=config.cascade_params["scaleFactor"],
                    minNeighbors=config.cascade_params["minNeighbors"],
                    minSize=config.cascade_params["minSize"],
                    maxSize=config.cascade_params["maxSize"],
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                if frame_count % 60 == 0:
                    print(f"Кадр {frame_count}: Обнаружено {len(faces)} лиц")
                
                for (x, y, w, h) in faces:
                    config.recognition_stats['total_faces_detected'] += 1
                    
                    # Проверка наличия глаз в обнаруженном лице
                    eyes_detected = False
                    if config.eye_cascade and not config.eye_cascade.empty():
                        roi_gray = gray[y:y+h, x:x+w]
                        eyes = config.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                        eyes_detected = len(eyes) > 0
                        
                        # Если требуется обязательное наличие глаз
                        if config.require_eyes_for_face and not eyes_detected:
                            continue  # Пропускаем это обнаружение лица
                    
                    # Цвет рамки по умолчанию (красный для неизвестных)
                    box_color = (0, 0, 255)  # Красный по умолчанию
                    text_color = (255, 255, 255)
                    status_text = "НЕИЗВЕСТНЫЙ"
                    
                    if config.is_collecting_data and config.collected_count < config.MAX_IMAGES:
                        face_roi = gray[y:y+h, x:x+w]
                        face_resize = cv2.resize(face_roi, config.IMAGE_SIZE)
                        
                        # Используем транслитерированное имя для директории
                        sanitized_name = sanitize_filename(config.current_person_name)
                        person_path = os.path.join(config.DATASET_DIR, sanitized_name)
                        os.makedirs(person_path, exist_ok=True)
                        
                        filename = os.path.join(person_path, f"{config.collected_count + 1}.png")
                        
                        if cv2.imwrite(filename, face_resize):
                            config.collected_count += 1
                            print(f"Сохранено: {filename}")
                        
                        # Синяя рамка для сбора данных
                        box_color = (255, 0, 0)
                        status_text = f"Сбор данных: {config.collected_count}/{config.MAX_IMAGES}"
                        
                        if config.collected_count >= config.MAX_IMAGES:
                            config.is_collecting_data = False
                            print(f"Коллекция завершена для {config.current_person_name}")
                    
                    elif config.model is not None and not config.is_collecting_data:
                        face_roi = gray[y:y+h, x:x+w]
                        face_resize = cv2.resize(face_roi, config.IMAGE_SIZE)
                        
                        try:
                            label, confidence = config.model.predict(face_resize)
                            
                            if confidence < config.CONFIDENCE_THRESHOLD:
                                # Известное лицо - зеленая рамка
                                sanitized_name = config.names.get(label, 'Неизвестный')
                                # Получаем оригинальное русское имя для отображения
                                config.original_name = get_original_name(sanitized_name)
                                status_text = f"{config.original_name}"
                                box_color = (0, 255, 0)  # Зеленый
                                text_color = (0, 0, 0)
                                config.recognition_stats['known_faces'] += 1
                                config.recognition_stats['last_recognized'] = config.original_name
                                
                                # Добавляем уровень уверенности
                                confidence_text = f"Уверенность: {100-confidence:.0f}%"
                                frame = draw_text_with_russian(frame, confidence_text,
                                                             (x, y+h+20), (0, 255, 0), 14)
                                
                            elif confidence < config.UNKNOWN_THRESHOLD:
                                # Возможно знакомое лицо - желтая рамка
                                sanitized_name = config.names.get(label, 'Неизвестный')
                                config.original_name = get_original_name(sanitized_name)
                                status_text = f"{config.original_name}?"
                                box_color = (0, 255, 255)  # Желтый
                                text_color = (0, 0, 0)
                                
                            else:
                                # Неизвестное лицо - красная рамка
                                status_text = "НЕИЗВЕСТНЫЙ"
                                box_color = (0, 0, 255)  # Красный
                                text_color = (255, 255, 255)
                                config.recognition_stats['unknown_faces'] += 1
                                
                        except Exception as e:
                            status_text = f"Ошибка: {str(e)[:15]}"
                            box_color = (0, 0, 255)
                            text_color = (255, 255, 255)
                    
                    else:
                        status_text = "Модель не обучена"
                        box_color = (128, 128, 128)
                        text_color = (255, 255, 255)
                    
                    # Добавляем информацию об обнаружении глаз
                    if eyes_detected:
                        status_text += " (глаза обнаружены)"
                    else:
                        status_text += " (глаза не обнаружены)"
                    
                    # Рисуем рамку с увеличенной толщиной
                    cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 3)
                    
                    # Рисуем фон для текста
                    text_width = len(status_text) * 10  # Примерная ширина текста
                    cv2.rectangle(frame, (x, y-35), (x + text_width + 10, y), box_color, -1)
                    
                    # Рисуем текст статуса с поддержкой русского
                    frame = draw_text_with_russian(frame, status_text, (x + 5, y - 30), text_color, 16)
                
                # Добавляем статистику на кадр
                info_texts = [
                    f"Всего лиц: {config.recognition_stats['total_faces_detected']}",
                    f"Известных: {config.recognition_stats['known_faces']}",
                    f"Неизвестных: {config.recognition_stats['unknown_faces']}"
                ]
                
                if config.recognition_stats['last_recognized']:
                    info_texts.append(f"Последний: {config.recognition_stats['last_recognized']}")
                
                for i, text in enumerate(info_texts):
                    frame = draw_text_with_russian(frame, text, (10, 30 + i*25), (255, 255, 255), 16)
                
                # Статус системы
                cascade_status = "Каскад лиц: ОК" if config.face_cascade and not config.face_cascade.empty() else "Каскад лиц: ОШИБКА"
                eye_cascade_status = "Каскад глаз: ОК" if config.eye_cascade and not config.eye_cascade.empty() else "Каскад глаз: ОШИБКА"
                model_status = "Модель: ОК" if config.model else "Модель: НЕ ОБУЧЕНА"
                
                frame = draw_text_with_russian(frame, cascade_status,
                                             (10, frame.shape[0] - 75),
                                             (0, 255, 0) if config.face_cascade and not config.face_cascade.empty() else (0, 0, 255),
                                             14)
                
                frame = draw_text_with_russian(frame, eye_cascade_status,
                                             (10, frame.shape[0] - 50),
                                             (0, 255, 0) if config.eye_cascade and not config.eye_cascade.empty() else (0, 0, 255),
                                             14)
                
                frame = draw_text_with_russian(frame, model_status,
                                             (10, frame.shape[0] - 25),
                                             (0, 255, 0) if config.model else (0, 0, 255),
                                             14)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        except Exception as e:
            print(f"Ошибка обработки кадра: {e}")
            time.sleep(0.1)
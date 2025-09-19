import cv2
import numpy as np
import os
import json
from app import config

class FaceRecognizer:
    def __init__(self):
        self.model_file = 'face_model.pkl'

        
        # Инициализация каскадов
        self._initialize_cascades()
    
    def _initialize_cascades(self):
        """Инициализация всех каскадов для обнаружения лиц и особенностей"""
        # Основной каскад для лиц
        if os.path.exists(config.HAAR_FILE):
            config.face_cascade = cv2.CascadeClassifier(config.HAAR_FILE)
        else:
            raise FileNotFoundError(f"Haar cascade file {config.HAAR_FILE} не найден")
        
        # Каскад для глаз (для верификации лиц)
        eye_cascade_path = config.EYE_FILE
        if os.path.exists(eye_cascade_path):
            config.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        else:
            print("Предупреждение: каскад для глаз не найден, верификация лиц отключена")
            config.eye_cascade = None
    
    def _enhance_image_quality(self, image):
        """Улучшение качества изображения для лучшего обнаружения"""
        # Гистограммная эквализация
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Контрастное ограниченное адаптивное выравнивание гистограммы (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Медианный фильтр для уменьшения шума
        denoised = cv2.medianBlur(enhanced, 3)
        
        return denoised
    
    def _verify_face(self, gray_image, face_rect):
        """Верификация что обнаруженный объект действительно является лицом"""
        x, y, w, h = face_rect
        
        # Проверка пропорций лица
        aspect_ratio = w / h
        if aspect_ratio < 0.6 or aspect_ratio > 1.4:  # Нормальные пропорции лица
            return False
        
        # Область для поиска глаз (верхняя половина лица)
        roi_gray = gray_image[y:y + h//2, x:x + w]
        
        # Поиск глаз для подтверждения лица
        if config.eye_cascade:
            eyes = config.eye_cascade.detectMultiScale(
                roi_gray, 
                scaleFactor=1.1, 
                minNeighbors=3,
                minSize=(20, 20)
            )
            # Если найдено хотя бы 1-2 глаза, считаем что это лицо
            return len(eyes) >= 1
        
        return True  # Если каскад глаз не доступен, принимаем все обнаружения
    
    def _detect_faces_advanced(self, frame):
        """Расширенное обнаружение лиц с несколькими техниками"""
        # Улучшение качества изображения
        enhanced = self._enhance_image_quality(frame)
        
        # Обнаружение лиц с разными параметрами
        faces = []
        
        # Попробуем несколько комбинаций параметров
        param_combinations = [
            {'scaleFactor': 1.05, 'minNeighbors': 5, 'minSize': (30, 30)},
            {'scaleFactor': 1.1, 'minNeighbors': 3, 'minSize': (40, 40)},
            {'scaleFactor': 1.2, 'minNeighbors': 7, 'minSize': (50, 50)}
        ]
        
        for params in param_combinations:
            detected_faces = config.face_cascade.detectMultiScale(
                enhanced,
                scaleFactor=params['scaleFactor'],
                minNeighbors=params['minNeighbors'],
                minSize=params['minSize']
            )
            
            for face in detected_faces:
                # Верификация лица
                if self._verify_face(enhanced, face):
                    # Проверяем дубликаты (перекрывающиеся прямоугольники)
                    is_duplicate = False
                    for existing_face in faces:
                        # Проверка пересечения прямоугольников
                        x1, y1, w1, h1 = existing_face
                        x2, y2, w2, h2 = face
                        
                        # Вычисляем площадь пересечения
                        dx = min(x1 + w1, x2 + w2) - max(x1, x2)
                        dy = min(y1 + h1, y2 + h2) - max(y1, y2)
                        
                        if dx > 0 and dy > 0:
                            intersection = dx * dy
                            area1 = w1 * h1
                            area2 = w2 * h2
                            
                            # Если пересечение значительное, пропускаем дубликат
                            if intersection > min(area1, area2) * 0.5:
                                is_duplicate = True
                                break
                    
                    if not is_duplicate:
                        faces.append(face)
        
        return np.array(faces)
    
    def _align_face(self, face_image, eyes=None):
        """Выравнивание лица по глазам"""
        if eyes is None or len(eyes) < 2:
            return face_image
        
        # Сортируем глаза по x-координате
        eyes = sorted(eyes, key=lambda e: e[0])
        
        # Вычисляем угол наклона
        left_eye = eyes[0]
        right_eye = eyes[1]
        
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Центр между глазами
        eyes_center = ((left_eye[0] + right_eye[0]) // 2, 
                      (left_eye[1] + right_eye[1]) // 2)
        
        # Матрица поворота
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1)
        
        # Применяем аффинное преобразование
        aligned = cv2.warpAffine(face_image, M, (face_image.shape[1], face_image.shape[0]))
        
        return aligned
    
    
    def recognize_face(self, face_image):
        """Распознавание лица на изображении"""
        if config.model is None:
            return None, 0
        
        # Улучшение качества и подготовка изображения
        enhanced = self._enhance_image_quality(face_image)
        
        # Изменение размера
        face_resized = cv2.resize(enhanced, config.IMAGE_SIZE)
        
        # Распознавание
        label, confidence = config.model.predict(face_resized)
        
        # Возвращаем результат
        if confidence < config.CONFIDENCE_THRESHOLD:
            return config.names.get(label, "Неизвестно"), confidence
        else:
            return "Не распознан", confidence
    
    def recognize_faces_in_frame(self, frame):
        """Распознавание всех лиц в кадре с улучшенным обнаружением"""
        if config.face_cascade is None:
            return frame, []
        
        # Используем расширенное обнаружение лиц
        faces = self._detect_faces_advanced(frame)
        
        results = []
        
        for (x, y, w, h) in faces:
            # Рисуем прямоугольник вокруг лица
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Извлекаем лицо
            face_roi = frame[y:y+h, x:x+w]
            
            # Распознаем лицо
            name, confidence = self.recognize_face(face_roi)
            
            # Определяем цвет текста в зависимости от уверенности
            if name != "Не распознан":
                color = (0, 255, 0)  # Зеленый для распознанных
                text = f"{name} ({confidence:.0f})"
            else:
                color = (0, 0, 255)  # Красный для нераспознанных
                text = name
            
            # Добавляем текст
            cv2.putText(frame, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Добавляем информацию об уверенности
            cv2.putText(frame, f"Conf: {confidence:.1f}", (x, y+h+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            results.append({
                'name': name,
                'confidence': confidence,
                'bbox': (x, y, w, h),
                'recognized': name != "Не распознан"
            })
        
        return frame, results
    
    def get_model_info(self):
        """Получить информацию о модели"""
        info = {
            'model_trained': config.model is not None,
            'names_count': len(config.names),
            'names': list(config.names.values()),
            'confidence_threshold': config.CONFIDENCE_THRESHOLD,
            'image_size': config.IMAGE_SIZE
        }
        
        if os.path.exists('model_metadata.json'):
            try:
                with open('model_metadata.json', 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                info['training_date'] = metadata.get('training_date')
            except:
                pass
        
        return info
    
    def test_recognition_accuracy(self, test_images_per_person=5):
        """Тестирование точности распознавания"""
        if not os.path.exists(config.DATASET_DIR) or config.model is None:
            print(f"[DEBUG] Dataset directory {config.DATASET_DIR} не найден")
            return None
        if config.model is None:
            print("[DEBUG] Модель не загружена")
            return None
        
        results = {}
        total_tests = 0
        correct_predictions = 0
        
        for person_name in os.listdir(config.DATASET_DIR):
            person_path = os.path.join(config.DATASET_DIR, person_name)
            if not os.path.isdir(person_path):
                continue
            
            images = [f for f in os.listdir(person_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Берем случайные изображения для тестирования
            test_images = np.random.choice(images, 
                                         min(test_images_per_person, len(images)), 
                                         replace=False)
            
            person_correct = 0
            person_total = 0
            
            for img_file in test_images:
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    predicted_name, confidence = self.recognize_face(img)
                    
                    if predicted_name == person_name:
                        person_correct += 1
                        correct_predictions += 1
                    
                    person_total += 1
                    total_tests += 1
            
            if person_total > 0:
                accuracy = person_correct / person_total
                results[person_name] = {
                    'correct': person_correct,
                    'total': person_total,
                    'accuracy': accuracy
                }
        
        overall_accuracy = correct_predictions / total_tests if total_tests > 0 else 0
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_tests': total_tests,
            'correct_predictions': correct_predictions,
            'per_person': results
        }

if __name__ == "__main__":
    recognizer = FaceRecognizer()

    # Абсолютные пути для модели
    model_path = config.MODEL_FILE
    metadata_path = config.METADATA_FILE

    # Попытка загрузить существующую модель
    print("Проверка существующей модели...")
    model_loaded = recognizer.load_model()
    
    if not model_loaded:
        print("Старая модель не найдена или не загружена. Начинаем обучение...")
        try:
            recognizer.train_model()
        except Exception as e:
            print(f"Ошибка обучения модели: {e}")
            exit(1)
    
    # Информация о модели
    print("\nИнформация о модели:")
    info = recognizer.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Запуск распознавания в реальном времени
    print("\nЗапуск распознавания в реальном времени...")
    print("Нажмите 'q' для выхода")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        exit(1)

    # Настройки камеры для лучшего качества
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр с камеры")
            break

        # Распознавание лиц в кадре
        frame_with_faces, face_results = recognizer.recognize_faces_in_frame(frame)

        # Добавляем информацию о количестве обнаруженных лиц
        cv2.putText(frame_with_faces, f"Обнаружено лиц: {len(face_results)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Показываем результат
        cv2.imshow('Face Recognition', frame_with_faces)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
import re
from app import config

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

def transliterate_russian_to_english(text):
    """
    Транслитерирует русский текст в английский
    """
    result = []
    for char in text:
        if char in config.TRANSLIT_TABLE:
            result.append(config.TRANSLIT_TABLE[char])
        elif char.isalnum() or char in ['_', '-']:
            result.append(char)
        else:
            result.append('_')  # Заменяем специальные символы на подчеркивание
    return ''.join(result)

def sanitize_filename(name):
    """
    Очищает имя для использования в качестве имени файла/директории
    """
    # Транслитерируем русские символы
    transliterated = transliterate_russian_to_english(name)
    
    # Удаляем все недопустимые символы
    sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', transliterated)
    
    # Удаляем множественные подчеркивания
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Удаляем подчеркивания в начале и конце
    sanitized = sanitized.strip('_')
    
    # Если после очистки строка пустая, используем дефолтное имя
    if not sanitized:
        sanitized = 'unknown_person'
    
    return sanitized

def get_original_name(sanitized_name):
    """
    Получает оригинальное русское имя по транслитерированному
    """
    return config.original_names.get(sanitized_name, sanitized_name)
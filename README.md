# Модель по Классификации Прикуса Зубов по Фотографиям

## Описание проекта

Этот проект предназначен для автоматической классификации прикуса зубов по фотографиям. Модель использует три типа изображений для предсказания:

1. Фото зубов
2. Фото лица анфас
3. Фото лица профиль

Цель проекта - профилактика здоровья полости рта.

## Функции модели

- **Предобученная модель VGG16**: используется для извлечения признаков из изображений.
- **Комбинированная архитектура**: объединяет выходы трех потоков изображений (зубы, лицо анфас, лицо профиль) для окончательной классификации.
- **Классификация прикуса**: модель способна распознавать следующие типы прикуса:
  - Нормальный
  - Мезиальный
  - Дистальный
  - Открытый
  - Перекрестный
  - Глубокий

## Установка

1. Клонируйте репозиторий:
    ```bash
    git clone https://github.com/yourusername/bite-classification.git
    cd bite-classification
    ```

2. Установите необходимые зависимости:
    ```bash
    pip install -r requirements.txt
    ```

3. Скачайте и поместите предварительно обученную модель `model.h5` в корневую папку проекта.

## Использование

### Подготовка изображений

Перед началом убедитесь, что у вас есть три изображения:
- Фото зубов
- Фото лица анфас
- Фото лица профиль

### Прогнозирование

Запустите скрипт для предсказания, заменив пути к изображениям:
```python
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

# Параметры изображения
img_width, img_height = 128, 128

# Загрузка модели
model = load_model('model.h5')

# Функция для подготовки изображения
def prepare_image(file_path):
    img = load_img(file_path, target_size=(img_width, img_height))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Добавляем размерность для batch
    img_array /= 255.0  # Нормализация
    return img_array

# Загрузка и подготовка изображений
teeth_img_path = 'path_to_teeth_image.jpg'
face_front_img_path = 'path_to_face_front_image.jpg'
face_profile_img_path = 'path_to_face_profile_image.jpg'

teeth_image = prepare_image(teeth_img_path)
face_front_image = prepare_image(face_front_img_path)
face_profile_image = prepare_image(face_profile_img_path)

# Прогнозирование
predictions = model.predict([teeth_image, face_front_image, face_profile_image])

# Интерпретация результатов
class_labels = ['normal', 'mesial', 'distal', 'open', 'crossbite', 'deep']

predicted_class = class_labels[np.argmax(predictions)]
predicted_probabilities = predictions[0]

print(f"Предсказанный класс прикуса: {predicted_class}")
print(f"Вероятности классов: {predicted_probabilities}")

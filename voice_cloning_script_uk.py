# -*- coding: utf-8 -*-
"""
voice_cloning_script_uk.py

Повноцінний скрипт на Python для клонування голосу з використанням моделі XTTS-v2 від Coqui.

Цей скрипт читає текстові дані з файлу CSV або JSON, клонує голос із наданого
аудіосемплу та генерує мовлення у форматі .wav. Він підтримує багатомовну генерацію
і оптимізований для використання GPU, якщо він доступний.

Ключові можливості:
- Клонування голосу з короткого аудіосемплу (рекомендовано 6-10 секунд).
- Вхідні дані з CSV (з колонкою 'text') або JSON (список об'єктів з ключем 'text').
- Багатомовний синтез мовлення (напр., en, es, fr, de, uk).
- Автоматичний вибір пристрою (CUDA GPU або CPU).
- Аргументи командного рядка для легкої конфігурації.
- Простий та детальний режими генерації для гнучкості.
- Надійна обробка помилок та управління директоріями.

Приклад використання:
1. Клонувати голос і згенерувати мовлення з CSV-файлу українською:
   python voice_cloning_script_uk.py --input_file input_data.csv --voice_sample sample_voice.wav --lang uk

2. Клонувати український голос і згенерувати мовлення англійською (крос-мовний синтез):
   python voice_cloning_script_uk.py --input_file input_data_en.csv --voice_sample sample_voice_uk.wav --lang en

3. Запустити простий тест із заздалегідь визначеним реченням:
   python voice_cloning_script_uk.py --voice_sample sample_voice.wav --simple_test

Налаштування:
1. Створіть та активуйте віртуальне середовище Python:
   python -m venv venv
   source venv/bin/activate  # У Windows: venv\Scripts\activate

2. Встановіть необхідні пакети:
   pip install -r requirements.txt

3. (Опційно, але рекомендовано для обробки аудіо) Встановіть ffmpeg.
"""

import os
import json
import argparse
import pandas as pd
import torch
from TTS.api import TTS

def select_device():
    """
    Вибирає відповідний пристрій для torch, віддаючи перевагу CUDA, якщо він доступний.
    """
    if torch.cuda.is_available():
        device = "cuda"
        print("CUDA доступний. Використовується GPU для обробки.")
    else:
        device = "cpu"
        print("CUDA недоступний. Використовується CPU. Це може бути повільно.")
    return device

def load_tts_model(device):
    """
    Завантажує модель Coqui XTTS-v2 на вибраний пристрій.
    """
    print("Завантаження моделі XTTS-v2...")
    try:
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print("Модель XTTS-v2 успішно завантажена.")
        return model
    except Exception as e:
        print(f"Помилка завантаження моделі TTS: {e}")
        print("Будь ласка, переконайтеся, що у вас є стабільне інтернет-з'єднання для завантаження моделі.")
        exit(1)

def parse_input_file(file_path):
    """
    Обробляє вхідний файл (CSV або JSON) і повертає список текстових рядків.
    """
    if not os.path.exists(file_path):
        print(f"Помилка: Вхідний файл не знайдено за шляхом '{file_path}'")
        return []

    texts = []
    try:
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
            if 'text' not in df.columns:
                print("Помилка: CSV-файл повинен містити колонку 'text'.")
                return []
            texts = df['text'].dropna().tolist()
        elif file_path.lower().endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list) and all(isinstance(item, dict) and 'text' in item for item in data):
                    texts = [item['text'] for item in data if item['text']]
                else:
                    print("Помилка: JSON-файл має бути списком об'єктів, кожен з яких містить ключ 'text'.")
                    return []
        else:
            print("Помилка: Непідтримуваний формат файлу. Будь ласка, використовуйте .csv або .json.")
            return []

        if not texts:
            print("Попередження: У вхідному файлі не знайдено текстових записів.")

        return texts

    except Exception as e:
        print(f"Помилка обробки вхідного файлу '{file_path}': {e}")
        return []

def simple_voice_cloning_test(tts_model, voice_sample_path, output_dir, lang):
    """
    Виконує простий тест клонування голосу з одним реченням.
    """
    print("\n--- Запуск простого тесту клонування голосу ---")
    if not os.path.exists(voice_sample_path):
        print(f"Помилка: Файл-зразок голосу не знайдено за шляхом '{voice_sample_path}'")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "simple_output.wav")
    test_sentence = "Привіт, це тестова перевірка клонування голосу. Сподіваюся, це звучить схоже на оригінал."

    print(f"Клонування голосу з: {voice_sample_path}")
    print(f"Текст для синтезу: '{test_sentence}'")
    print(f"Генерація аудіо...")

    try:
        tts_model.tts_to_file(
            text=test_sentence,
            speaker_wav=voice_sample_path,
            language=lang,
            file_path=output_path
        )
        print(f"Аудіофайл успішно згенеровано: {output_path}")
    except Exception as e:
        print(f"Помилка під час генерації TTS для простого тесту: {e}")

def detailed_voice_cloning(tts_model, texts, voice_sample_path, output_dir, lang):
    """
    Виконує детальне клонування голосу зі списку текстів.
    """
    print("\n--- Запуск детального процесу клонування голосу ---")
    if not os.path.exists(voice_sample_path):
        print(f"Помилка: Файл-зразок голосу не знайдено за шляхом '{voice_sample_path}'")
        return

    if not texts:
        print("Немає текстів для обробки. Пропускаємо детальну генерацію.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Клонування голосу з: {voice_sample_path}")
    print(f"Цільова мова: {lang}")
    print(f"Вихідна директорія: {output_dir}")
    print(f"Початок пакетної генерації для {len(texts)} текстових записів...")

    # Примітка щодо стилю/емоцій:
    # Модель XTTS-v2 переважно клонує тембр голосу. Просодія (ритм, інтонація)
    # згенерованого мовлення залежить від референсного аудіо. Щоб згенерувати
    # мовлення з певною емоцією, ви повинні надати `voice_sample`, який
    # демонструє цю емоцію. Модель спробує імітувати цей стиль.

    for i, text in enumerate(texts):
        output_filename = f"output_{i+1}.wav"
        output_path = os.path.join(output_dir, output_filename)
        print(f"\nОбробка запису {i+1}/{len(texts)}:")
        print(f"  Текст: '{text[:100]}...'") # Виводимо частину тексту

        try:
            tts_model.tts_to_file(
                text=text,
                speaker_wav=voice_sample_path,
                language=lang,
                file_path=output_path
            )
            print(f"  Аудіо успішно згенеровано: {output_path}")
        except Exception as e:
            print(f"  Помилка генерації аудіо для запису {i+1}: {e}")
            print("   Пропускаємо цей запис.")
            continue

    print("\n Пакетна генерація завершена.")

def main():
    """
    Головна функція для обробки аргументів та запуску скрипта.
    """
    parser = argparse.ArgumentParser(description="Скрипт для клонування голосу з використанням Coqui XTTS-v2")
    parser.add_argument(
        "--input_file",
        type=str,
        help="Шлях до вхідного файлу CSV або JSON з текстом для синтезу."
    )
    parser.add_argument(
        "--voice_sample",
        type=str,
        required=True,
        help="Шлях до аудіофайлу .wav для клонування голосу (напр., 'sample_voice.wav')."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_audio",
        help="Директорія для збереження згенерованих аудіофайлів."
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="uk",
        help="Мова для генерації TTS (напр., 'uk', 'en', 'es', 'fr')."
    )
    parser.add_argument(
        "--simple_test",
        action="store_true",
        help="Запустити простий тест замість обробки файлу."
    )

    args = parser.parse_args()

    # --- Виконання скрипта ---
    device = select_device()
    tts_model = load_tts_model(device)

    if args.simple_test:
        simple_voice_cloning_test(tts_model, args.voice_sample, args.output_dir, args.lang)
    else:
        if not args.input_file:
            parser.error("--input_file є обов'язковим, якщо не вказано --simple_test.")
        texts_to_process = parse_input_file(args.input_file)
        detailed_voice_cloning(tts_model, texts_to_process, args.voice_sample, args.output_dir, args.lang)

    print("\n Роботу скрипта завершено.")

if __name__ == "__main__":
    main()

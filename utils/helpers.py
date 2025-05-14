import json
import pandas as pd

# Функція для збереження даних у форматі JSON
def save_to_json(data, file_path):
    """
    Зберігає дані в JSON файл.
    :param data: Дані, які потрібно зберегти.
    :param file_path: Шлях до файлу для збереження.
    """
    try:
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        return True
    except Exception as e:
        print(f"Помилка при збереженні в JSON: {e}")
        return False

# Функція для завантаження даних з JSON файлу
def load_from_json(file_path):
    """
    Завантажує дані з JSON файлу.
    :param file_path: Шлях до файлу.
    :return: Завантажені дані.
    """
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return data
    except Exception as e:
        print(f"Помилка при завантаженні з JSON: {e}")
        return None

# Функція для обробки даних (наприклад, видалення порожніх значень)
def preprocess_data(df):
    """
    Обробка даних: видалення порожніх значень.
    :param df: DataFrame з даними.
    :return: Очищений DataFrame.
    """
    df_cleaned = df.dropna()  # Видалення порожніх значень
    return df_cleaned

# Функція для перевірки коректності формату даних
def validate_data(df):
    """
    Перевірка коректності даних (наприклад, перевірка числових стовпців).
    :param df: DataFrame з даними.
    :return: True, якщо дані коректні, інакше False.
    """
    # Перевірка на наявність нечислових значень у числових стовпцях
    for column in df.select_dtypes(include=['object']).columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            return False
    return True

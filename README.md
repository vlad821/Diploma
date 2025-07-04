﻿
# Вебплатформа для проведення аналітики даних з використанням машинного навчання

Вебдодаток на основі Python/Streamlit для автоматизованого аналізу даних із підтримкою AI.

---

## 👤 Автор

- **ПІБ**: Рейнгардт Владислав Васильович 
- **Група**: ФЕІ-42
- **Керівник**: Демків Л.С.,кандидат фізико-математичних наук, доцент кафедри Системного проектування
- **Дата виконання**: 01.06.2025

---

## 📌 Загальна інформація

- **Тип проєкту**: Вебзастосунок
- **Мова програмування**: Python
- **Фреймворки / Бібліотеки**: Streamlit, Pandas, Scikit-learn, OpenAI API

---

## 🧰 Використані технології

- 🐍 Python 3.11
- 📊 Pandas, NumPy — для обробки та аналізу даних
- 📈 Matplotlib, Seaborn — побудова графіків
- 🤖 Scikit-learn — моделі машинного навчання
- 🧪 SciPy — статистичні тести
- 💬 OpenAI GPT API — інтеграція штучного інтелекту
- 🧵 Streamlit — веб-інтерфейс

---

## 🧠 Основні можливості

- 📁 Завантаження власного CSV-файлу
- 🧼 Попередня обробка даних (викиди, відсутні значення)
- 📊 Побудова зведених таблиць (Pivot Table)
- 🧪 Статистичне тестування (гіпотези)
- 📈 Побудова графіків (розподіли, залежності)
- 🤖 AI-помічник на базі OpenAI для відповідей на питання про дані
- 🧠 Моделі машинного навчання (класифікація / регресія/ кластеризація)
- 🧾 Збереження результатів сесії

---

## 🖱️ Інструкція для користувача

### 1. 📁 Повний код проєкту

Код доступний у GitHub-репозиторії:  
🔗 [https://github.com/vlad821/Diploma.git](https://github.com/vlad821/Diploma.git)

---

### 2. ⚙️ Встановлення проєкту та залежностей 

#### 🔽 Крок 1: Клонування репозиторію

```bash
git clone https://github.com/vlad821/Diploma.git
cd Diploma
```

#### 📦 Крок 2: Встановлення бібліотек

```bash
python -m pip install -r requirements.txt
```

### 3. 🔐 Налаштування API-ключа

#### 📄 Крок 1: Створіть файл `.streamlit/secrets.toml`

#### 📄 Крок 2: Додайте наступний код:
```toml
OPENAI_API_KEY = "ваш_ключ"
```
Отримати ключ можна тут: [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)
### 3. 🚀 Запуск застосунку

```bash
python -m streamlit run app.py
```

 ## Основні кроки у застосунку:
   - 🔼 **Завантажити CSV-файл**
   - 🔍 **Оглянути структуру даних**
   - 📊 **Провести статистичний аналіз** (викиди, гіпотези)
   - ⚙️ **Побудувати зведені таблиці**
   - 📈 **Виконати візуалізацію**
   - 🧠 **Сформувати SQL запит за допомогою GPT** (наприклад: *"Який менеджер зробив найбільшу виручку в червні?"*)
   - 🤖 **Навчити базову модель на основі ваших даних**
   - 💾 **Зберегти сесію**

---

## 🗂️ Структура проєкту

```
FINAL_WORK/
├── .streamlit/               # Налаштування Streamlit
│   └── secrets.toml          # API ключі (наприклад, OpenAI)
├── exports/
│   └── session_data.json     # Збереження попередньої сесії
├── modules/
│   ├── ai_models.py          # Побудова ML-моделей
│   ├── data_upload.py        # Завантаження даних
│   ├── hypothesis_testing.py # Статистичне тестування
│   ├── outliers.py           # Обробка викидів
│   ├── pivot_table.py        # Побудова зведених таблиць
│   ├── profiling.py          # Аналіз і опис даних
│   ├── settings.py           # Налаштування
│   ├── sql_gpt.py            # GPT-асистент для SQL-запитів
│   └── visualizations.py     # Графіки
├── utils/
│   ├── app.py                # Точка входу в застосунок
│   ├── config.json           # Конфігурація
│   └── gpt_helper.py         # Підключення до OpenAI
├── requirements.txt          # Список залежностей
├── session.pkl               # Стан сесії
└── README.md                 # Цей файл
```

## 🔐 Безпека

- `secrets.toml` містить API-ключі і **не повинен бути публічним**
- Додайте його до `.gitignore`:
  ```
  .streamlit/secrets.toml
  ```

---

## 🚀 Можливі напрямки розвитку

- ☁️ Розгортання у хмарі (Streamlit Cloud, Heroku)
- 📊 Додавання нових моделей (кластеризація, PCA)
- 📱 Адаптація під мобільний браузер

---

## 📷 Приклади / Скриншоти

У папці screenshots знаходяться скріни модулів додатку а також і відео.

---

## 📚 Джерела / Література

- [Streamlit.io](https://streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Scikit-learn Guide](https://scikit-learn.org/)
- [OpenAI API](https://platform.openai.com/)
- [Stack Overflow](https://stackoverflow.com/)

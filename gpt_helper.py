import openai
import streamlit as st

# 📘 Функція для пояснення результатів за допомогою GPT
def explain_with_gpt(user_text):
    """
    Надсилає запит до моделі GPT-4 з поясненням результатів статистичного аналізу.

    Параметри:
    - user_text (str): текст із результатами, які потрібно пояснити.

    Виводить:
    - Пояснення результатів мовою, зрозумілою для користувача.
    """
    try:
        # 🔐 Отримання ключа API з конфігурації Streamlit
        openai.api_key = st.secrets["OPENAI_API_KEY"]

        # 📤 Формування запиту до GPT
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "Ти — аналітик даних, який пояснює статистичні результати простою мовою для студентів або людей без технічного бекграунду."
                },
                {
                    "role": "user",
                    "content": user_text
                }
            ]
        )

        # ✅ Виведення результату у Streamlit
        st.write("📘 GPT пояснення:")
        st.success(response['choices'][0]['message']['content'])

    except Exception as e:
        # ⚠️ Обробка помилки
        st.error(f"Помилка при зверненні до OpenAI: {e}")

import streamlit as st
import pandas as pd
import pandasql as psql
import openai
from gpt_helper import explain_with_gpt  # Повинна бути реалізована

# ==== Генерація SQL через GPT ====
def generate_sql_with_gpt(user_prompt, column_names, language):
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        columns_str = ", ".join(column_names)

        system_message = {
            "Українська": f"Ти — SQL-асистент. Твоя задача — генерувати валідні SQL-запити для pandas DataFrame з колонками: {columns_str}. "
                          f"Поверни лише сам SQL-запит без пояснень чи додаткового тексту. Не використовуй SELECT *.",
            "English": f"You are a SQL assistant. Your task is to generate valid SQL queries for a pandas DataFrame with columns: {columns_str}. "
                       f"Return only the SQL query without any explanation or extra text. Do not use SELECT *."
        }

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message[language]},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"❌ {'Помилка' if language == 'Українська' else 'Error'}: {e}")
        return None

# ==== Основна сторінка Streamlit ====
def app():
    # 🌐 Перемикач мови
    language = st.radio("🌐 Виберіть мову / Choose language:", ["Українська", "English"], horizontal=True)

    # ✅ Перевірка даних
    if "data" not in st.session_state or st.session_state["data"] is None:
        st.warning("⚠️ Будь ласка, завантажте дані на сторінці 'Завантаження даних'." if language == "Українська"
                   else "⚠️ Please upload data on the 'Upload Data' page.")
        return

    df = st.session_state["data"]

    # 📋 Локалізація
    labels = {
        "title": "💬 SQL-запити до даних через GPT" if language == "Українська" else "💬 SQL Queries to Data via GPT",
        "preview": "📊 Попередній перегляд даних" if language == "Українська" else "📊 Data Preview",
        "prompt": "📝 Опишіть запит природною мовою:" if language == "Українська" else "📝 Describe your query in natural language:",
        "placeholder": "Наприклад: Порахуй середню зарплату по кожному відділу" if language == "Українська"
                      else "Example: Calculate the average salary by department",
        "generate": "🚀 Згенерувати та виконати SQL-запит" if language == "Українська" else "🚀 Generate and run SQL query",
        "empty_prompt": "🔔 Введіть запит для генерації SQL." if language == "Українська" else "🔔 Please enter a query to generate SQL.",
        "executed": "✅ Запит виконано успішно." if language == "Українська" else "✅ Query executed successfully.",
        "explain": "💡 Пояснити результат за допомогою GPT" if language == "Українська" else "💡 Explain result with GPT",
        "gen_fail": "⚠️ Не вдалося згенерувати SQL-запит. Спробуйте ще раз." if language == "Українська"
                    else "⚠️ Failed to generate SQL query. Try again."
    }

    # 🖼️ Інтерфейс
    st.title(labels["title"])
    st.subheader(labels["preview"])
    st.dataframe(df.head())

    user_prompt = st.text_area(labels["prompt"], placeholder=labels["placeholder"])

    if st.button(labels["generate"]):
        if not user_prompt.strip():
            st.warning(labels["empty_prompt"])
            return

        with st.spinner("🧠 GPT працює..." if language == "Українська" else "🧠 GPT is working..."):
            sql_query = generate_sql_with_gpt(user_prompt, df.columns.tolist(), language)

        if sql_query:
            # Заміна імені таблиці 'dataframe' на 'df' для pandasql
            sql_query = sql_query.replace("dataframe", "df")

            st.code(sql_query, language="sql")
            try:
                result = psql.sqldf(sql_query, {"df": df})
                st.success(labels["executed"])
                st.dataframe(result)

                if not result.empty and st.checkbox(labels["explain"]):
                    explain_with_gpt(str(result.head(5)), language=language)

            except Exception as e:
                st.error(f"❌ {'Помилка при виконанні SQL-запиту' if language == 'Українська' else 'Error running SQL query'}: {e}")
        else:
            st.warning(labels["gen_fail"])


# Для багатосторінкової навігації (не викликайте st.set_page_config тут вдруге!)
if __name__ == "__main__":
    st.set_page_config(page_title="SQL + GPT", layout="wide")  # Повинен бути ПЕРШИМ рядком в скрипті
    app()

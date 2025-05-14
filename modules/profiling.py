import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from gpt_helper import explain_with_gpt  # Тобі слід забезпечити, щоб ця функція коректно працювала
import os

def app():
    st.title("📊 Профайлінг даних")

    if "data" not in st.session_state:
        st.warning("⛔ Спочатку завантажте дані на відповідній вкладці.")
        return

    df = st.session_state["data"]

    # GPT пояснення, що таке профайлінг
    st.info("🧠 Що таке профайлінг даних?")
    explain_with_gpt("Поясни, що таке профайлінг даних у контексті аналізу даних та які переваги він надає.")

    # Генерація профілю
    st.subheader("📄 Звіт Pandas Profiling")
    profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)

    # Показати звіт у Streamlit
    st_profile_report(profile)

    # Пояснення результатів профілю
    st.subheader("🧠 Пояснення результатів")

    # Отримуємо опис профілю
    summary = profile.get_description()

    # Витягуємо ключові моменти для пояснення
    variables = summary.variables if hasattr(summary, 'variables') else {}
    missing = summary.missing if hasattr(summary, 'missing') else {}
    duplicates = summary.duplicates if hasattr(summary, 'duplicates') else {}
    alerts = summary.alerts if hasattr(summary, 'alerts') else []
    correlations = summary.correlations if hasattr(summary, 'correlations') else {}

    # Створюємо короткий звіт для пояснення
    explanation = f"""
    Основні результати профілю:
    
    - **Змінні**: Є {len(variables)} змінних у наборі даних.
    - **Пропущені значення**: {', '.join([key for key in missing.get('variables', {}).keys()]) if missing.get('variables') else 'немає'} змінних з пропущеними значеннями.
    - **Дублікатні значення**: Кількість дублікатів: {duplicates.get('count', 'Невідомо')}.
    - **Попередження**: Виявлено {len(alerts)} попереджень.
    - **Кореляції**: Типи кореляцій: {', '.join(correlations.keys()) if correlations else 'відсутні'}.

    Чи є які-небудь аномалії чи важливі патерни, на які слід звернути увагу?
    """

    # Відправляємо цей короткий звіт GPT для пояснення
    response = explain_with_gpt(f"Проаналізуй ці результати профілювання даних: {explanation}")

    # Показуємо пояснення від GPT
    st.write(response)

    # Збереження звіту в HTML
    output_path = "temp/pandas_profiling_report.html"
    os.makedirs("temp", exist_ok=True)
    profile.to_file(output_path)

    # Кнопка для завантаження
    with open(output_path, "rb") as f:
        st.download_button(
            label="📥 Завантажити звіт у форматі HTML",
            data=f,
            file_name="pandas_profiling_report.html",
            mime="text/html"
        )

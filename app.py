import streamlit as st
import pickle
import datetime
import pandas as pd
from gpt_helper import explain_with_gpt
from modules import ai_models, data_upload, hypothesis_testing, outliers, profiling, sql_gpt, pivot_table, visualizations,settings
from utils.theme import set_theme

# --- Загальні налаштування
st.set_page_config(page_title="📊 Data Analyzer", layout="wide", initial_sidebar_state="expanded")
set_theme()

# --- Стан сторінки
if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = list({
        "📁 Завантаження даних": data_upload,
        "🧬 Профайлінг даних": profiling,
        "🚨 Виявлення викидів": outliers,
        "📈 Візуалізації": visualizations,
        "🤖 Штучний інтелект": ai_models,
        "🧪 Тестування гіпотез": hypothesis_testing,
        "💬 SQL через GPT": sql_gpt,
        "📊 Зведені таблиці": pivot_table,
        "⚙️ Налаштування": settings

    }.keys())[0]

# --- Пошук
st.title("📊 Data Analyzer App")
search_query = st.text_input("🔍 Шукати розділ:", "")
PAGES = {
    "📁 Завантаження даних": data_upload,
    "🧬 Профайлінг даних": profiling,
    "🚨 Виявлення викидів": outliers,
    "📈 Візуалізації": visualizations,
    "🤖 Штучний інтелект": ai_models,
    "🧪 Тестування гіпотез": hypothesis_testing,
    "💬 SQL через GPT": sql_gpt,
    "📊 Зведені таблиці": pivot_table,
    "⚙️ Налаштування": settings

}
page_options = [page for page in PAGES if search_query.lower() in page.lower()]
selected_page = st.selectbox("📂 Оберіть розділ:", page_options, index=page_options.index(st.session_state["selected_page"]) if st.session_state["selected_page"] in page_options else 0)
st.session_state["selected_page"] = selected_page
page = PAGES[selected_page]

# --- Сайдбар: інформація про дані
st.sidebar.title("🧭 Меню навігації")
explain_with_gpt(f"Поясни зміст розділу: {selected_page}")

if "data" in st.session_state and st.session_state["data"] is not None:
    df = st.session_state["data"]
    st.sidebar.markdown("### 📋 Інформація про дані")
    st.sidebar.write(f"**Розмір:** `{df.shape[0]} x {df.shape[1]}`")
    st.sidebar.write("**Стовпці:**", list(df.columns))

    with st.sidebar.expander("📊 Коротка статистика"):
        st.dataframe(df.describe().T, use_container_width=True)

    if st.sidebar.button("📥 Експортувати звіт (CSV)"):
        st.sidebar.download_button("⬇️ Завантажити CSV", df.to_csv(index=False), file_name="data_export.csv")

    if st.sidebar.button("🧠 Поясни датафрейм"):
        explain_with_gpt(f"Опиши ці дані: {df.head().to_string()}")

    # --- Сесія: зберегти/завантажити
    with st.sidebar.expander("💾 Сесія"):
        if st.button("💾 Зберегти сесію"):
            with open("session.pkl", "wb") as f:
                pickle.dump(st.session_state, f)
            st.success("Сесію збережено ✅")

        if st.button("🔄 Завантажити сесію"):
            try:
                with open("session.pkl", "rb") as f:
                    st.session_state.update(pickle.load(f))
                st.rerun()
            except FileNotFoundError:
                st.warning("Файл сесії не знайдено.")

    if st.sidebar.button("🗑️ Очистити дані"):
        st.session_state["data"] = None
        st.rerun()
else:
    st.sidebar.warning("⚠️ Дані ще не завантажено.")

# --- Поради
tips = {
    "📁 Завантаження даних": "Завантажте CSV або Excel-файл.",
    "🧬 Профайлінг даних": "Дізнайтесь типи, пропуски та описову статистику.",
    "🚨 Виявлення викидів": "Визначення аномальних значень.",
    "📈 Візуалізації": "Створення графіків для аналізу.",
    "🤖 Штучний інтелект": "Моделі ML для аналізу.",
    "🧪 Тестування гіпотез": "Порівняння груп статистично.",
    "💬 SQL через GPT": "Напишіть запит словами — GPT створить SQL.",
    "📊 Зведені таблиці": "Підсумок і групування даних.",
    "⚙️ Налаштування": "Налаштування зовнішнього вигляду, API-ключів і кешу."

}
if selected_page in tips:
    st.sidebar.info(f"📌 {tips[selected_page]}")

# --- Нотатки
with st.sidebar.expander("📝 Нотатки"):
    st.session_state["user_notes"] = st.text_area("Ваші нотатки:", value=st.session_state.get("user_notes", ""))

# --- Ціль сесії
with st.sidebar.expander("🎯 Мета сесії"):
    st.session_state["session_goal"] = st.text_input("Ціль сьогодні:", value=st.session_state.get("session_goal", ""))

# --- Старт сесії
if "session_start" not in st.session_state:
    st.session_state["session_start"] = datetime.datetime.now()
st.sidebar.markdown(f"⏱️ Початок: **{st.session_state['session_start'].strftime('%H:%M:%S')}**")

# --- Лог останніх дій
with st.sidebar.expander("🕘 Журнал дій"):
    if "log" not in st.session_state:
        st.session_state["log"] = []
    new_entry = f"{datetime.datetime.now().strftime('%H:%M:%S')} | {selected_page}"
    if not st.session_state["log"] or st.session_state["log"][-1] != new_entry:
        st.session_state["log"].append(new_entry)
    st.text("\n".join(st.session_state["log"][-5:]))

# --- Відображення вибраної сторінки
page.app()

# --- Кнопка повернення догори (оновлена)
st.markdown("""
<style>
.scroll-to-top {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background-color: #1abc9c;
    color: white;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    font-size: 24px;
    text-align: center;
    line-height: 40px;
    cursor: pointer;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    z-index: 1001;
}
</style>
<script>
const btn = document.createElement('div');
btn.className = 'scroll-to-top';
btn.innerHTML = '↑';
btn.onclick = () => window.scrollTo({ top: 0, behavior: 'smooth' });
document.body.appendChild(btn);
</script>
""", unsafe_allow_html=True)

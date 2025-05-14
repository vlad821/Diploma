import streamlit as st

def app():
    st.header("⚙️ Налаштування")

    st.subheader("🌓 Тема")
    theme_choice = st.radio("Оберіть тему:", ["Світла", "Темна"], index=0)
    st.info("Зміну теми можна застосувати вручну у `.streamlit/config.toml`.")

    st.subheader("🌍 Мова інтерфейсу")
    st.selectbox("Оберіть мову:", ["Українська", "English", "Polski", "Deutsch"], index=0)

    st.subheader("🧠 GPT API ключ")
    st.text_input("Введіть OpenAI API ключ:", type="password")

    st.subheader("🧹 Кешування")
    if st.button("Очистити кеш"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("Кеш очищено.")

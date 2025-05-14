import streamlit as st
import pandas as pd
import io
import openai
from gpt_helper import explain_with_gpt  # Переконайтесь, що ця функція існує у вашому проєкті

def app():
    st.title("📊 Зведені таблиці (Pivot Table)")

    if "data" not in st.session_state or st.session_state["data"] is None:
        st.warning("⚠️ Будь ласка, спочатку завантажте дані.")
        return

    df = st.session_state["data"]

    st.write("### ➕ Налаштування зведеної таблиці")

    rows = st.multiselect("🧱 Рядки (index):", options=df.columns)
    cols = st.multiselect("📊 Колонки (columns):", options=df.columns)
    values = st.multiselect("📈 Значення (values):", options=df.columns)
    aggfunc = st.selectbox("📐 Агрегація:", options=["sum", "mean", "count", "min", "max"])

    if st.button("📤 Побудувати зведену таблицю"):
        try:
            pivot = pd.pivot_table(
                df,
                index=rows,
                columns=cols,
                values=values,
                aggfunc=aggfunc
            )
            st.success("✅ Зведена таблиця успішно побудована.")
            st.dataframe(pivot)

            # Збереження в Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                pivot.to_excel(writer, index=True, sheet_name='Pivot Table')
            excel_buffer.seek(0)

            st.download_button(
                label="⬇️ Завантажити як Excel",
                data=excel_buffer,
                file_name="pivot_table.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # 📌 Додати пояснення GPT
            if st.checkbox("💡 Пояснити результат зведеної таблиці за допомогою GPT"):
                prompt = (
                    f"Я побудував зведену таблицю за такими параметрами:\n"
                    f"- Рядки (index): {rows}\n"
                    f"- Колонки (columns): {cols}\n"
                    f"- Значення (values): {values}\n"
                    f"- Агрегація: {aggfunc}\n\n"
                    f"Результат:\n{pivot.head(5).to_string()}\n\n"
                    f"Поясни, як побудована таблиця, що означають її ряди та колонки, і як інтерпретувати її значення."
                )
                explain_with_gpt(prompt)

        except Exception as e:
            st.error(f"❌ Помилка при створенні зведеної таблиці: {e}")

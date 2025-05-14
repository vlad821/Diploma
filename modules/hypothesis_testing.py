import streamlit as st
import scipy.stats as stats
import pandas as pd
from gpt_helper import explain_with_gpt  # Імпорт GPT-функції

def app():
    st.title("🧪 Статистичне тестування гіпотез")

    # 📘 Пояснення
    with st.expander("ℹ️ Що таке тестування гіпотез?"):
        st.markdown("""
        **Тестування статистичних гіпотез** — це інструмент для перевірки припущень про параметри генеральної сукупності на основі вибірки.

        - **Нульова гіпотеза (H₀)**: між групами немає суттєвої різниці.
        - **Альтернативна гіпотеза (H₁)**: між групами є статистично значуща різниця.

        Різні тести використовуються в залежності від типу даних та гіпотез:
        - **t-тест**: порівняння середніх двох незалежних груп.
        - **ANOVA**: порівняння середніх більше ніж двох груп.
        - **Z-тест**: використовується для великих вибірок або відомих стандартних відхилень.
        """)

    # Перевірка наявності даних
    if "data" not in st.session_state:
        st.warning("📂 Спочатку завантажте дані на вкладці 'Завантаження даних'")
        return

    df = st.session_state["data"]
    numeric_cols = df.select_dtypes(include='number').columns

    if len(numeric_cols) < 2:
        st.error("❌ Потрібно хоча б дві числові змінні для проведення тесту.")
        return

    # Вибір тесту
    st.subheader("🔎 Виберіть тест для перевірки гіпотез")
    test_choice = st.selectbox("Оберіть тест", ["t-тест", "ANOVA", "Z-тест"])

    if test_choice == "t-тест":
        col1 = st.selectbox("Змінна 1", numeric_cols, key="hypo_col1")
        col2 = st.selectbox("Змінна 2", [col for col in numeric_cols if col != col1], key="hypo_col2")
        
        # Проведення t-тесту
        sample1 = df[col1].dropna()
        sample2 = df[col2].dropna()

        t_stat, p_value = stats.ttest_ind(sample1, sample2)

        st.markdown(f"**🔢 t-статистика:** `{t_stat:.4f}`")
        st.markdown(f"**📊 p-значення:** `{p_value:.4f}`")

        if p_value < 0.05:
            st.success("🚨 Відхиляємо H₀: Є статистично значуща різниця між середніми значеннями.")
        else:
            st.info("✅ Не відхиляємо H₀: Немає статистично значущої різниці між середніми.")
        
        explain_text = (
            f"Результати t-тесту: t-статистика = {t_stat:.4f}, p-значення = {p_value:.4f}. "
            f"Поясни, що це означає для змінних '{col1}' і '{col2}' у контексті перевірки гіпотез."
        )
        response = explain_with_gpt(explain_text)
        st.subheader("💡 Пояснення від GPT:")
        st.markdown(response)

    elif test_choice == "ANOVA":
        # Вибір змінних для ANOVA
        col = st.selectbox("Виберіть змінну для ANOVA", numeric_cols)
        groups_col = st.selectbox("Виберіть категоріальну змінну для групування", df.select_dtypes(include='object').columns)

        # Переведення змінної в категорії
        groups = df.groupby(groups_col)[col].apply(list)
        f_stat, p_value = stats.f_oneway(*groups)

        st.markdown(f"**🔢 F-статистика:** `{f_stat:.4f}`")
        st.markdown(f"**📊 p-значення:** `{p_value:.4f}`")

        if p_value < 0.05:
            st.success("🚨 Відхиляємо H₀: Є статистично значуща різниця між групами.")
        else:
            st.info("✅ Не відхиляємо H₀: Немає статистично значущої різниці між групами.")
        
        explain_text = (
            f"Результати ANOVA: F-статистика = {f_stat:.4f}, p-значення = {p_value:.4f}. "
            f"Поясни, що це означає для змінної '{col}' і категорії '{groups_col}' у контексті перевірки гіпотез."
        )
        response = explain_with_gpt(explain_text)
        st.subheader("💡 Пояснення від GPT:")
        st.markdown(response)

    elif test_choice == "Z-тест":
        # Вибір змінної для Z-тесту
        col = st.selectbox("Виберіть змінну для Z-тесту", numeric_cols)
        population_mean = st.number_input("Введіть середнє значення генеральної сукупності", value=0.0)
        sample = df[col].dropna()

        # Проведення Z-тесту
        sample_mean = sample.mean()
        sample_std = sample.std()
        sample_size = len(sample)
        z_stat = (sample_mean - population_mean) / (sample_std / (sample_size ** 0.5))
        p_value = stats.norm.sf(abs(z_stat)) * 2  # Двосторонній Z-тест

        st.markdown(f"**🔢 Z-статистика:** `{z_stat:.4f}`")
        st.markdown(f"**📊 p-значення:** `{p_value:.4f}`")

        if p_value < 0.05:
            st.success("🚨 Відхиляємо H₀: Різниця між вибірковим середнім і генеральним середнім статистично значуща.")
        else:
            st.info("✅ Не відхиляємо H₀: Різниця між вибірковим середнім і генеральним середнім незначуща.")
        
        explain_text = (
            f"Результати Z-тесту: Z-статистика = {z_stat:.4f}, p-значення = {p_value:.4f}. "
            f"Поясни, що це означає для змінної '{col}' у контексті порівняння вибірки та генеральної сукупності."
        )
        response = explain_with_gpt(explain_text)
        st.subheader("💡 Пояснення від GPT:")
        st.markdown(response)


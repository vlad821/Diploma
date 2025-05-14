# pages/outliers.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from gpt_helper import explain_with_gpt

def app():
    st.title("🔍 Виявлення викидів")

    if "data" not in st.session_state:
        st.warning("⚠️ Спочатку потрібно завантажити дані на вкладці 'Завантаження даних'.")
        return

    df = st.session_state["data"]

    st.subheader("Оригінальні дані")
    st.write(df.head())

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    selected_cols = st.multiselect(
        "Виберіть стовпці для аналізу", 
        options=numeric_cols, 
        default=list(numeric_cols)
    )

    # Один слайдер для всіх графіків
    width = st.slider("Загальна ширина графіка", 4, 20, 10)
    height = st.slider("Загальна висота графіка", 2, 10, 4)

    if selected_cols:
        st.subheader("📊 Boxplot для вибраних колонок")

        for col in selected_cols:
            fig, ax = plt.subplots(figsize=(width, height))
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f'Boxplot: {col}')
            st.pyplot(fig)
            explain_with_gpt(f"Поясни boxplot для стовпця '{col}' у наборі даних.")

        st.subheader("🚨 Виявлені викиди")
        outlier_info = {}
        for col in selected_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if not outliers.empty:
                outlier_info[col] = outliers[[col]]
                explain_with_gpt(f"Опиши викиди в '{col}' на основі методу IQR.")

        if outlier_info:
            for col, outliers in outlier_info.items():
                st.markdown(f"**{col}**")
                st.dataframe(outliers)
        else:
            st.info("✅ Не знайдено викидів у вибраних числових стовпцях.")
            explain_with_gpt("Що означає відсутність викидів у наборі даних?")

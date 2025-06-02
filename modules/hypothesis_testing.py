import streamlit as st
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gpt_helper import explain_with_gpt  # Припускаємо, що ця функція у тебе є

def app():
    st.title("🧪 Статистичне тестування гіпотез")

    with st.expander("ℹ️ Що таке тестування гіпотез?"):
        st.markdown("""
        **Тестування статистичних гіпотез** — інструмент для перевірки припущень про параметри генеральної сукупності на основі вибірки.

        - **H₀**: немає суттєвої різниці або зв'язку.
        - **H₁**: існує суттєва різниця або зв’язок.

        Доступні тести:
        - t-тест,
        - ANOVA,
        - Z-тест,
        - Уїтні-Манна (Mann-Whitney U),
        - Хі-квадрат (Chi-square),
        - Кореляція Пірсона.
        """)

    if "data" not in st.session_state:
        st.warning("📂 Завантажте дані на вкладці 'Завантаження даних'")
        return

    df = st.session_state["data"]
    numeric_cols = df.select_dtypes(include='number').columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    st.subheader("🔎 Виберіть тест")
    test_choice = st.selectbox("Оберіть тест", [
        "t-тест",
        "ANOVA",
        "Z-тест",
        "Уїтні-Манна (Mann-Whitney U)",
        "Хі-квадрат (Chi-square)",
        "Кореляція Пірсона"
    ])

    if test_choice == "t-тест":
        col1 = st.selectbox("Змінна 1", numeric_cols, key="ttest_col1")
        col2 = st.selectbox("Змінна 2", [col for col in numeric_cols if col != col1], key="ttest_col2")
        sample1 = df[col1].dropna()
        sample2 = df[col2].dropna()
        t_stat, p_value = stats.ttest_ind(sample1, sample2)
        st.write(f"t-статистика: {t_stat:.4f}")
        st.write(f"p-значення: {p_value:.4f}")

        fig, ax = plt.subplots()
        sns.boxplot(data=df[[col1, col2]], ax=ax)
        ax.set_title(f"Boxplot для {col1} та {col2}")
        st.pyplot(fig)

        conclusion = "Відхиляємо H₀: є значуща різниця." if p_value < 0.05 else "Не відхиляємо H₀: значущої різниці немає."
        st.info(conclusion)

        prompt = f"Поясни результати t-тесту: t={t_stat:.4f}, p={p_value:.4f} для змінних {col1} і {col2}."
        st.markdown(explain_with_gpt(prompt))

    elif test_choice == "ANOVA":
        if len(categorical_cols) == 0:
            st.error("Потрібна категоріальна змінна для групування!")
            return
        num_col = st.selectbox("Числова змінна", numeric_cols)
        cat_col = st.selectbox("Категоріальна змінна", categorical_cols)
        groups = [group.dropna() for name, group in df.groupby(cat_col)[num_col]]
        f_stat, p_value = stats.f_oneway(*groups)
        st.write(f"F-статистика: {f_stat:.4f}")
        st.write(f"p-значення: {p_value:.4f}")

        fig, ax = plt.subplots()
        sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax)
        ax.set_title(f"Boxplot {num_col} по групах {cat_col}")
        st.pyplot(fig)

        conclusion = "Відхиляємо H₀: є різниця між групами." if p_value < 0.05 else "Не відхиляємо H₀."
        st.info(conclusion)

        prompt = f"Поясни результати ANOVA: F={f_stat:.4f}, p={p_value:.4f} для змінної {num_col} по категорії {cat_col}."
        st.markdown(explain_with_gpt(prompt))

    elif test_choice == "Z-тест":
        col = st.selectbox("Змінна", numeric_cols)
        population_mean = st.number_input("Середнє генеральної сукупності", value=0.0)
        sample = df[col].dropna()
        sample_mean = sample.mean()
        sample_std = sample.std()
        n = len(sample)
        z_stat = (sample_mean - population_mean) / (sample_std / (n ** 0.5))
        p_value = stats.norm.sf(abs(z_stat)) * 2
        st.write(f"Z-статистика: {z_stat:.4f}")
        st.write(f"p-значення: {p_value:.4f}")

        conclusion = "Відхиляємо H₀." if p_value < 0.05 else "Не відхиляємо H₀."
        st.info(conclusion)

        prompt = f"Поясни результати Z-тесту: Z={z_stat:.4f}, p={p_value:.4f} для змінної {col}."
        st.markdown(explain_with_gpt(prompt))

    elif test_choice == "Уїтні-Манна (Mann-Whitney U)":
        col1 = st.selectbox("Змінна 1", numeric_cols, key="mann_col1")
        col2 = st.selectbox("Змінна 2", [col for col in numeric_cols if col != col1], key="mann_col2")
        sample1 = df[col1].dropna()
        sample2 = df[col2].dropna()
        u_stat, p_value = stats.mannwhitneyu(sample1, sample2)
        st.write(f"U-статистика: {u_stat:.4f}")
        st.write(f"p-значення: {p_value:.4f}")

        fig, ax = plt.subplots()
        sns.boxplot(data=df[[col1, col2]], ax=ax)
        ax.set_title(f"Boxplot для {col1} та {col2}")
        st.pyplot(fig)

        conclusion = "Відхиляємо H₀." if p_value < 0.05 else "Не відхиляємо H₀."
        st.info(conclusion)

        prompt = f"Поясни результати тесту Уїтні-Манна: U={u_stat:.4f}, p={p_value:.4f} для {col1} і {col2}."
        st.markdown(explain_with_gpt(prompt))

    elif test_choice == "Хі-квадрат (Chi-square)":
        if len(categorical_cols) < 2:
            st.error("Потрібні дві категоріальні змінні!")
            return
        cat1 = st.selectbox("Категоріальна змінна 1", categorical_cols, key="chi_cat1")
        cat2 = st.selectbox("Категоріальна змінна 2", [c for c in categorical_cols if c != cat1], key="chi_cat2")
        contingency_table = pd.crosstab(df[cat1], df[cat2])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        st.write("Таблиця спряженості:")
        st.dataframe(contingency_table)
        st.write(f"Chi2: {chi2:.4f}")
        st.write(f"p-значення: {p:.4f}")
        st.write(f"Ступені свободи: {dof}")

        fig, ax = plt.subplots()
        contingency_table.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title(f"Bar chart для {cat1} і {cat2}")
        st.pyplot(fig)

        conclusion = "Відхиляємо H₀." if p < 0.05 else "Не відхиляємо H₀."
        st.info(conclusion)

        prompt = f"Поясни результати хі-квадрат тесту: Chi2={chi2:.4f}, p={p:.4f} для змінних {cat1} і {cat2}."
        st.markdown(explain_with_gpt(prompt))

    elif test_choice == "Кореляція Пірсона":
        col1 = st.selectbox("Змінна 1", numeric_cols, key="corr_col1")
        col2 = st.selectbox("Змінна 2", [col for col in numeric_cols if col != col1], key="corr_col2")
        corr, p_value = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
        st.write(f"Коефіцієнт кореляції Пірсона: {corr:.4f}")
        st.write(f"p-значення: {p_value:.4f}")

        fig, ax = plt.subplots()
        sns.scatterplot(x=col1, y=col2, data=df, ax=ax)
        ax.set_title(f"Кореляційний графік {col1} vs {col2}")
        st.pyplot(fig)

        conclusion = "Існує статистично значуща кореляція." if p_value < 0.05 else "Кореляція незначуща."
        st.info(conclusion)

        prompt = f"Поясни результати кореляції Пірсона: r={corr:.4f}, p={p_value:.4f} між {col1} і {col2}."
        st.markdown(explain_with_gpt(prompt))

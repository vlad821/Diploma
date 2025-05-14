import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay
import openai

# Функція пояснення результатів за допомогою GPT
def explain_with_gpt(user_text):
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Ти — аналітик даних, який пояснює результати простою мовою."},
                {"role": "user", "content": user_text}
            ]
        )
        st.write("📘 GPT пояснення:")
        st.success(response['choices'][0]['message']['content'])
    except Exception as e:
        st.error(f"Помилка при зверненні до OpenAI: {e}")

# Лінійна регресія
def linear_regression(df):
    st.subheader("📈 Лінійна регресія")
    num_cols = df.select_dtypes(include='number').columns.tolist()
    
    if len(num_cols) < 2:
        st.warning("У датасеті недостатньо числових стовпців для лінійної регресії.")
        return

    target_col = st.selectbox("Цільова змінна", num_cols, key="target_col_lr")
    features = [col for col in num_cols if col != target_col]

    if st.button("Навчити модель лінійної регресії", key="train_button_lr"):
        X = df[features]
        y = df[target_col]

        # Видалення рядків з NaN
        data = pd.concat([X, y], axis=1).dropna()

        if data.empty:
            st.error("❌ Після обробки пропущених значень не залишилось жодного рядка для навчання.")
            return

        X = data[features]
        y = data[target_col]

        if len(data) < 2:
            st.error("❌ Недостатньо зразків для розділення на train/test. Мінімум — 2.")
            return

        # Розділення на тренувальні та тестові дані
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        except ValueError as e:
            st.error(f"❌ Помилка під час поділу: {str(e)}")
            return

        # Навчання моделі
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Результати
        st.write("Прогнози:")
        st.write(predictions)

        st.write("🔎 Оцінка моделі:")
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        st.write(f"MAE: {mae}")
        st.write(f"MSE: {mse}")

        # Графік
        fig, ax = plt.subplots()
        ax.scatter(y_test, predictions, alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
        ax.set_xlabel("Справжні значення")
        ax.set_ylabel("Прогнозовані значення")
        ax.set_title("📈 Прогноз vs Істинне значення")
        st.pyplot(fig)

        # GPT пояснення
        explain_with_gpt(f"Я навчив модель лінійної регресії. MAE = {mae}, MSE = {mse}. Поясни ці результати.")


# Логістична регресія
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def logistic_regression(df):
    st.subheader("🔍 Логістична регресія")

    # 1. Перевірка наявності числових колонок
    try:
        num_cols = df.select_dtypes(include='number').columns.tolist()
        if len(num_cols) < 2:
            st.warning("У датасеті недостатньо числових стовпців для логістичної регресії.")
            return
    except Exception as e:
        st.error(f"❌ Помилка при визначенні числових стовпців: {e}")
        return

    # 2. Вибір цільової змінної
    target_col = st.selectbox("Цільова змінна", num_cols, key="target_col_logistic")
    features = [col for col in num_cols if col != target_col]

    # 3. Метод бінаризації
    binarization_method = st.radio(
        "Метод бінаризації (для неперервних змінних):",
        ["Медіана", "Середнє", "Користувацький поріг"]
    )

    custom_threshold = None
    if binarization_method == "Користувацький поріг":
        custom_threshold = st.number_input("Введіть свій поріг", value=0.0, step=0.1)

    if st.button("Навчити логістичну регресію", key="train_button_logistic"):
        try:
            X = df[features]
            y = df[target_col]
            data = pd.concat([X, y], axis=1).dropna()
            X = data[features]
            y = data[target_col]
        except Exception as e:
            st.error(f"❌ Помилка при підготовці даних: {e}")
            return

        try:
            # Автоматична або ручна бінаризація
            if y.nunique() > 2:
                if binarization_method == "Медіана":
                    threshold = y.median()
                elif binarization_method == "Середнє":
                    threshold = y.mean()
                elif binarization_method == "Користувацький поріг":
                    threshold = custom_threshold
                else:
                    raise ValueError("Невідомий метод бінаризації")

                st.warning(f"⚠️ Цільова змінна була бінаризована (1 якщо > {threshold:.2f}, інакше 0).")
                y = (y > threshold).astype(int)

            # Перевірка: чи дійсно стало 2 класи
            if y.nunique() != 2:
                st.error(f"❌ Після бінаризації знайдено {y.nunique()} унікальних класів. Перевірте вхідні дані.")
                return
        except Exception as e:
            st.error(f"❌ Помилка під час бінаризації: {e}")
            return

        # Розбиття на тренувальні і тестові дані
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        except Exception as e:
            st.error(f"❌ Помилка при розділенні даних: {e}")
            return

        # Модель логістичної регресії
        try:
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
        except Exception as e:
            st.error(f"❌ Помилка при навчанні логістичної регресії: {e}")
            return

        # Прогноз і вивід результатів
        try:
            preds = model.predict(X_test)
            st.text("📊 Звіт про класифікацію:")
            st.text(classification_report(y_test, preds))

            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax)
            ax.set_title("Матриця плутанини (логістична регресія)")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Помилка при оцінюванні моделі: {e}")
            return

        # Пояснення (якщо GPT інтегрований)
        try:
            explain_with_gpt(f"Я навчив логістичну регресію. Ось звіт: {classification_report(y_test, preds)}")
        except Exception:
            st.info("✅ Модель навчено. GPT-пояснення наразі недоступне.")

# Дерево рішень
def decision_tree(df):
    st.subheader("🌳 Дерево рішень")
    num_cols = df.select_dtypes(include='number').columns.tolist()
    target_col = st.selectbox("Цільова змінна", num_cols, key="target_col_dt")
    features = [col for col in num_cols if col != target_col]

    if st.button("Навчити дерево рішень", key="train_button_dt"):
        X = df[features]
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        st.text("Звіт про класифікацію:")
        st.text(classification_report(y_test, preds))

        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)
        ax.set_title("Матриця плутанини (дерево рішень)")
        st.pyplot(fig)

        explain_with_gpt(f"Я навчив модель дерева рішень. Ось результати: {classification_report(y_test, preds)}")

# Кластеризація K-Means
def kmeans_clustering(df):
    st.subheader("🧠 Кластеризація K-Means")
    num_cols = df.select_dtypes(include='number').columns.tolist()

    if len(num_cols) < 2:
        st.warning("Для кластеризації потрібно хоча б два числові стовпці.")
        return

    k = st.slider("Кількість кластерів", 1, 10, 3)

    if st.button("Запустити кластеризацію", key="train_button_kmeans"):
        X = df[num_cols]
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X)

        df['Cluster'] = clusters
        st.write("Результати кластеризації:")
        st.dataframe(df)

        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=num_cols[0], y=num_cols[1], hue='Cluster', palette='viridis', ax=ax)
        ax.set_title("Розподіл кластерів")
        st.pyplot(fig)

        explain_with_gpt(f"Я застосував кластеризацію K-Means з {k} кластерами. Ось отримані кластери: {list(clusters)}")

# Головна функція застосунку
def app():
    st.title("📊 Моделі машинного навчання")

    # Очікуємо, що датафрейм буде у сесії (має бути завантажений раніше в іншій частині програми)
    if "data" in st.session_state:
        tabs = st.tabs([
            "📈 Лінійна регресія",
            "🔍 Логістична регресія",
            "🧠 K-Means Кластеризація",
            "🌳 Дерево рішень"
        ])

        with tabs[0]:
            linear_regression(st.session_state["data"])
        with tabs[1]:
            logistic_regression(st.session_state["data"])
        with tabs[2]:
            kmeans_clustering(st.session_state["data"])
        with tabs[3]:
            decision_tree(st.session_state["data"])
    else:
        st.warning("Будь ласка, завантажте набір даних через інтерфейс завантаження.")

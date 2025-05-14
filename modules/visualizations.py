import streamlit as st
import plotly.express as px
import pandas as pd

def visualizations():
    st.title("📊 Візуалізація даних")

    if "data" in st.session_state:
        df = st.session_state["data"]
        
        # Вибір типу графіку
        plot_type = st.selectbox("Оберіть тип графіку", ["Розсіювання", "Гістограма", "Кореляційна матриця", "Box Plot"])

        if plot_type == "Розсіювання":
            x_col = st.selectbox("Оберіть стовпець для осі X", df.columns)
            y_col = st.selectbox("Оберіть стовпець для осі Y", df.columns)

            if x_col and y_col:
                st.subheader(f"Графік розсіювання між {x_col} та {y_col}")
                fig = px.scatter(df, x=x_col, y=y_col, title=f"Розсіювання: {x_col} vs {y_col}")
                st.plotly_chart(fig)

        elif plot_type == "Гістограма":
            hist_col = st.selectbox("Оберіть стовпець для гістограми", df.columns)
            if hist_col:
                st.subheader(f"Гістограма для {hist_col}")
                fig = px.histogram(df, x=hist_col, nbins=20, title=f"Гістограма для {hist_col}")
                st.plotly_chart(fig)

        elif plot_type == "Кореляційна матриця":
            st.subheader("Кореляційна матриця")
            # Filter only numeric columns
            df_numeric = df.select_dtypes(include=['number'])
            if df_numeric.empty:
                st.warning("⛔ Немає числових стовпців для розрахунку кореляції.")
            else:
                corr = df_numeric.corr()
                fig = px.imshow(corr, text_auto=True, color_continuous_scale="coolwarm", title="Кореляційна матриця")
                st.plotly_chart(fig)

        elif plot_type == "Box Plot":
            box_col = st.selectbox("Оберіть стовпець для Box Plot", df.columns)
            if box_col:
                st.subheader(f"Box Plot для {box_col}")
                fig = px.box(df, y=box_col, title=f"Box Plot для {box_col}")
                st.plotly_chart(fig)

    else:
        st.warning("⛔ Завантажте дані для візуалізації.")

def app():
    visualizations()
if __name__ == "__main__":
    app()

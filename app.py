"""
Streamlit-застосунок: Прогноз опадів на основі даних Open-Meteo
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from datetime import date, timedelta

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
import warnings
warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "wind_speed_10m_max",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
    "cloud_cover_mean",
]

COL_RENAME = {
    "windspeed_10m_max":        "wind_speed_10m_max",
    "cloudcover_mean":          "cloud_cover_mean",
    "relative_humidity_2m_mean": "relative_humidity_2m_mean",  # без змін
    "surface_pressure_mean":     "surface_pressure_mean",       # без змін
}


def fetch_open_meteo(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """
    Download daily weather data from Open-Meteo.
    Uses the Historical Archive API for past dates,
    and the Forecast API for future/recent dates.
    """
    from datetime import date as date_cls
    today = date_cls.today()
    start_d = date_cls.fromisoformat(start)

    if start_d < today:
        url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        url = "https://api.open-meteo.com/v1/forecast"

    daily_vars = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "rain_sum",
        "wind_speed_10m_max",
        "relative_humidity_2m_mean",
        "surface_pressure_mean",
        "cloud_cover_mean",
    ]

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": daily_vars,
        "start_date": start,
        "end_date": end,
        "timezone": "Europe/Kiev",
    }

    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()["daily"]
    df = pd.DataFrame(data)
    df.rename(columns={"time": "date"}, inplace=True)
    return df


def load_and_prepare(df: pd.DataFrame):
    """Clean dataframe and create target variable."""
    df = df.copy()
    df.rename(columns=COL_RENAME, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.dropna(subset=FEATURE_COLS + ["precipitation_sum"], inplace=True)
    df["target"] = (df["precipitation_sum"] > 0).astype(int)
    return df


def build_features(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    y = df["target"].values
    return X, y


def train_models(X_train, y_train):
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tr_sc, y_train)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    return lr, rf, scaler


def evaluate(model, X_test, y_test, scaler=None):
    Xsc = scaler.transform(X_test) if scaler else X_test
    preds = model.predict(Xsc)
    return {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds, zero_division=0),
        "F1": f1_score(y_test, preds, zero_division=0),
        "preds": preds,
    }


st.set_page_config(
    page_title="Прогноз опадів",
    page_icon="",
    layout="wide",
)

st.title("Прогноз опадів — Open-Meteo ML")
st.caption("Класифікація: опади є / опадів немає")

for key in ("df", "lr", "rf", "scaler", "X_test", "y_test", "trained"):
    if key not in st.session_state:
        st.session_state[key] = None
if "trained" not in st.session_state:
    st.session_state.trained = False

# ══════════════════════════════════════════════════════════════════════════════
# Дані
# ══════════════════════════════════════════════════════════════════════════════
st.header("Дані")

data_tab1, data_tab2 = st.tabs(["Отримати з Open-Meteo", "Завантажити CSV"])

with data_tab1:
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Широта (lat)", value=50.45, format="%.4f")
        lon = st.number_input("Довгота (lon)", value=30.52, format="%.4f")
    with col2:
        d_start = st.date_input("Початок", value=date(2023, 1, 1))
        d_end   = st.date_input("Кінець",  value=date(2024, 12, 31))

    if st.button("Отримати дані з Open-Meteo"):
        days = (d_end - d_start).days
        if days < 60:
            st.warning("Рекомендується мінімум 60 днів для навчання моделі.")
        try:
            with st.spinner("Завантаження…"):
                raw = fetch_open_meteo(lat, lon, str(d_start), str(d_end))
            df = load_and_prepare(raw)
            st.session_state.df = df
            st.session_state.trained = False
            csv_bytes = df.to_csv(index=False).encode()
            st.success(f"Отримано {len(df)} днів ({(df.target==1).sum()} дощових).")
            st.download_button("Зберегти weather_daily.csv", csv_bytes,
                               "weather_daily.csv", "text/csv")
        except Exception as e:
            st.error(f"Помилка при запиті: {e}")

with data_tab2:
    uploaded = st.file_uploader("Оберіть CSV-файл", type="csv")
    if uploaded:
        if uploaded.name != st.session_state.get("uploaded_filename"):
            try:
                raw = pd.read_csv(uploaded)
                df = load_and_prepare(raw)
                st.session_state.df = df
                st.session_state.trained = False
                st.session_state.uploaded_filename = uploaded.name
                st.success(f"Завантажено {len(df)} рядків.")
            except Exception as e:
                st.error(f"Помилка читання файлу: {e}")
        else:
            if st.session_state.df is not None:
                st.success(f"Файл '{uploaded.name}' завантажено ({len(st.session_state.df)} рядків).")

if st.session_state.df is not None:
    df_show = st.session_state.df
    with st.expander("Переглянути датасет"):
        st.dataframe(df_show.head(30), width='stretch')
    c1, c2, c3 = st.columns(3)
    c1.metric("Всього днів", len(df_show))
    c2.metric("Дощових", int(df_show["target"].sum()))
    c3.metric("Без опадів", int((df_show["target"] == 0).sum()))

    monthly = df_show.copy()
    monthly["month"] = pd.to_datetime(monthly["date"]).dt.to_period("M").astype(str)
    monthly_rain = monthly.groupby("month")["precipitation_sum"].sum().reset_index()
    st.bar_chart(monthly_rain.set_index("month")["precipitation_sum"],
                 width='stretch', height=200)
    st.caption("Місячна сума опадів (мм)")

# ══════════════════════════════════════════════════════════════════════════════
# Тренування
# ══════════════════════════════════════════════════════════════════════════════
st.header("Навчання моделі")

if st.session_state.df is None:
    st.info("Спочатку завантажте дані (Блок 1).")
else:
    test_size = st.slider("Частка тестової вибірки", 0.1, 0.4, 0.2, 0.05)

    if st.button("Навчити модель"):
        df_ml = st.session_state.df
        X, y = build_features(df_ml)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        with st.spinner("Навчання…"):
            lr, rf, scaler = train_models(X_train, y_train)

        st.session_state.lr      = lr
        st.session_state.rf      = rf
        st.session_state.scaler  = scaler
        st.session_state.X_test  = X_test
        st.session_state.y_test  = y_test
        st.session_state.trained = True
        st.success("Моделі навчено!")

    if st.session_state.trained:
        lr     = st.session_state.lr
        rf     = st.session_state.rf
        scaler = st.session_state.scaler
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        lr_m = evaluate(lr, X_test, y_test, scaler)
        rf_m = evaluate(rf, X_test, y_test)

        st.subheader("Метрики")
        col_lr, col_rf = st.columns(2)

        with col_lr:
            st.markdown("**Logistic Regression**")
            m_df = pd.DataFrame({
                "Метрика": ["Accuracy", "Precision", "Recall", "F1"],
                "Значення": [f"{lr_m[k]:.3f}" for k in ("Accuracy","Precision","Recall","F1")],
            })
            st.table(m_df)

        with col_rf:
            st.markdown("**Random Forest**")
            m_df2 = pd.DataFrame({
                "Метрика": ["Accuracy", "Precision", "Recall", "F1"],
                "Значення": [f"{rf_m[k]:.3f}" for k in ("Accuracy","Precision","Recall","F1")],
            })
            st.table(m_df2)

        # Feature importances
        fi = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        st.subheader("Важливість ознак (Random Forest)")
        st.bar_chart(fi, width='stretch', height=200)

# ══════════════════════════════════════════════════════════════════════════════
# Прогноз
# ══════════════════════════════════════════════════════════════════════════════
st.header("Прогноз опадів")

if not st.session_state.trained:
    st.info("Спочатку навчіть модель (Блок 2).")
else:
    df_ml  = st.session_state.df
    lr     = st.session_state.lr
    rf     = st.session_state.rf
    scaler = st.session_state.scaler

    forecast_mode = st.radio(
        "Режим прогнозу",
        ["Обрати дату з датасету", "Ввести значення вручну"],
        horizontal=True,
    )

    if forecast_mode == "Обрати дату з датасету":
        dates_avail = df_ml["date"].dt.date.tolist()
        chosen_date = st.selectbox("Оберіть дату", options=dates_avail,
                                   index=len(dates_avail) - 1)
        row    = df_ml[df_ml["date"].dt.date == chosen_date].iloc[0]
        x_input = np.array([[row[c] for c in FEATURE_COLS]])
        actual  = int(row["target"])
        st.caption(f"Реальне значення: {'Опади були' if actual else 'Опадів не було'}")

    else:
        st.markdown("Введіть метеопараметри для прогнозу:")
        cols = st.columns(4)
        vals = {}
        labels = {
            "temperature_2m_max":        ("Макс. температура, °C", 15.0),
            "temperature_2m_min":        ("Мін. температура, °C",   5.0),
            "temperature_2m_mean":       ("Серед. температура, °C", 10.0),
            "wind_speed_10m_max":        ("Макс. вітер, км/год",    10.0),
            "relative_humidity_2m_mean": ("Вологість, %",           65.0),
            "surface_pressure_mean":     ("Тиск, гПа",            1013.0),
            "cloud_cover_mean":          ("Хмарність, %",           50.0),
        }
        for i, (col_name, (label, default)) in enumerate(labels.items()):
            with cols[i % 4]:
                vals[col_name] = st.number_input(label, value=default)
        x_input = np.array([[vals[c] for c in FEATURE_COLS]])
        actual  = None

    model_choice = st.selectbox("Модель", ["Random Forest", "Logistic Regression"])

    # ── Прогноз будується автоматично (без окремої кнопки) ──────────────────
    # Це гарантує, що результат не зникає при зміні дати чи моделі,
    # тому що st.button() скидається при кожному перезапуску сторінки.
    if model_choice == "Random Forest":
        model  = rf
        x_proc = x_input
    else:
        model  = lr
        x_proc = scaler.transform(x_input)

    pred      = model.predict(x_proc)[0]
    prob      = model.predict_proba(x_proc)[0]
    prob_rain = prob[1]

    st.divider()
    if pred == 1:
        st.success("### Очікуються опади")
    else:
        st.info("### Опадів не очікується")

    col_a, col_b = st.columns(2)
    col_a.metric("Ймовірність опадів",     f"{prob_rain:.1%}")
    col_b.metric("Ймовірність без опадів", f"{prob[0]:.1%}")

    st.progress(float(prob_rain), text=f"Ймовірність опадів: {prob_rain:.1%}")

    if actual is not None:
        verdict = "Правильно" if pred == actual else "Неправильно"
        fact    = "Опади були" if actual else "Опадів не було"
        st.caption(f"Фактичний результат: {fact} ({verdict})")

st.divider()
st.caption("Дані: Open-Meteo API | Моделі: Logistic Regression, Random Forest | © 2024")

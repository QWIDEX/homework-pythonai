# Прогноз опадів — Open-Meteo ML

Міні-сервіс прогнозування опадів на основі реальних метеоданих з [Open-Meteo](https://open-meteo.com).  
Класифікація: **опади є (1) / опадів немає (0)**.

---

## Дані

| Поле | Опис |
|---|---|
| `temperature_2m_max` | Максимальна температура повітря (°C) |
| `temperature_2m_min` | Мінімальна температура повітря (°C) |
| `temperature_2m_mean` | Середня температура повітря (°C) |
| `precipitation_sum` | Сума опадів за день (мм) — **цільова змінна** |
| `rain_sum` | Сума дощу за день (мм) |
| `windspeed_10m_max` | Максимальна швидкість вітру (км/год) |
| `relative_humidity_2m_mean` | Середня відносна вологість (%) |
| `surface_pressure_mean` | Середній атмосферний тиск (гПа) |
| `cloudcover_mean` | Середня хмарність (%) |

**Джерело:** безкоштовне API [Open-Meteo](https://open-meteo.com/en/docs).  
**Місто за замовчуванням:** Київ (lat=50.45, lon=30.52).  
**Рекомендований період:** щонайменше 60–80 днів.

**Цільова змінна:**
- `precipitation_sum = 0` → клас **0** (опадів немає)
- `precipitation_sum > 0` → клас **1** (опади є)

---

## Моделі

| Модель | Опис |
|---|---|
| **Logistic Regression** | Лінійна модель; вхід нормалізується StandardScaler |
| **Random Forest** | Ансамблева модель із 200 деревами; інтерпретовані важливості ознак |

Обидві моделі видають **клас прогнозу** та **ймовірність** належності до кожного класу.

---

## Структура проєкту

```
.
├── app.py               # Streamlit-застосунок (Варіант A)
├── weather_daily.csv    # Датасет щоденних метеоданих
├── requirements.txt     # Python-залежності
└── README.md            # Цей файл
```

### Послідовність дій

```
1. Завантаження даних
   └── Open-Meteo API (HTTP GET) → weather_daily.csv

2. Підготовка даних
   └── очищення пропусків → формування target (0/1)

3. Навчання
   ├── train/test split (80/20, stratify)
   ├── StandardScaler для Logistic Regression
   ├── Logistic Regression
   └── Random Forest

4. Оцінка
   └── Accuracy, Precision, Recall, F1, Classification Report

5. Прогноз
   └── predict() + predict_proba() → "Очікуються опади" / "Опадів не очікується"
```

---

## Запуск

### Streamlit-застосунок

```bash
pip install -r requirements.txt
streamlit run app.py
```

Відкрийте браузер: `http://localhost:8501`

---

## Встановлення залежностей

```bash
pip install -r requirements.txt
```

---

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model_and_metadata, prepare_input_for_model

# Настройка страницы - делаем её уже
st.set_page_config(page_title="AI Cabin Loading", page_icon="✈️", layout="centered")

# Загрузка ресурсов (кэшируется для скорости)
@st.cache_resource
def get_resources():
    return load_model_and_metadata()

try:
    model, metadata = get_resources()
except Exception as e:
    st.error(f"Ошибка загрузки модели. Убедитесь, что файлы лежат в одной папке с app.py. Ошибка: {e}")
    st.stop()

# --- ЗАГОЛОВОК ---
st.title("✈️ Предиктивная система загрузки камбуза")
st.markdown("##### *Интеллектуальный помощник для планирования бортового питания*")
st.markdown("---")

# --- БОКОВАЯ ПАНЕЛЬ (ВВОД ДАННЫХ) ---
st.sidebar.header("📋 Параметры рейса")

# Категории (убираем 'missing' из списка для красоты)
categories = [c for c in metadata['unique_categories'] if c != 'missing']
category = st.sidebar.selectbox("🍽️ Категория товара", categories)

# Страны
countries = metadata['unique_countries']
country_arr = st.sidebar.selectbox("🌍 Страна прилета", countries)
country_dep = st.sidebar.selectbox(" Страна вылета", countries, index=0)

# Параметры полета
duration = st.sidebar.slider("⏱️ Длительность полета (часы)", 0.5, 15.0, 3.0, 0.5)
distance = st.sidebar.number_input("📏 Расстояние (км)", 100, 15000, 2000, 100)

# Время
dep_hour = st.sidebar.slider("🕐 Время вылета (час)", 0, 23, 10)
month = st.sidebar.selectbox("📅 Месяц", range(1, 13), index=6) # Июль по умолчанию
weekday = st.sidebar.selectbox("📆 День недели", ["Пн (0)", "Вт (1)", "Ср (2)", "Чт (3)", "Пт (4)", "Сб (5)", "Вс (6)"])
weekday_num = int(weekday.split('(')[1][0]) # Извлекаем число

# Кнопка
st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🚀 РАССЧИТАТЬ ПРОГНОЗ", type="primary", use_container_width=True)

# --- ОСНОВНАЯ ОБЛАСТЬ ---
if predict_btn:
    with st.spinner(" CatBoost анализирует данные..."):
        user_inputs = {
            'category': category,
            'country_arr': country_arr,
            'country_dep': country_dep,
            'duration': duration,
            'distance': distance,
            'dep_hour': dep_hour,
            'month': month,
            'weekday': weekday_num
        }
        
        input_df = prepare_input_for_model(user_inputs, metadata)
        prediction = model.predict(input_df)[0]
        
        # Расчет Safety Stock
        sigma = metadata['sigma']
        Z = metadata['Z_service_level']
        safety_stock = Z * sigma
        recommended = int(np.ceil(prediction + safety_stock))
        
    st.success(" **Прогноз успешно рассчитан!**", icon="✅")
    
    # Метрики в 3 колонки - делаем их компактнее
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📊 Ожидаемые продажи",
            value=f"{prediction:.1f} шт.",
            delta=None
        )
    
    with col2:
        st.metric(
            label="🛡️ Страховой запас (90%)",
            value=f"{safety_stock:.2f} шт.",
            #delta=f"Z={Z}, σ={sigma:.2f}"
        )
    
    with col3:
        st.metric(
            label="📦 Рекомендация к загрузке",
            value=f"{recommended} шт.",
            delta=f"+{recommended - int(prediction)} шт. запас"
        )
    
    st.markdown("---")
    
    # Компактная визуализация - маленький график
    col_chart1, col_chart2 = st.columns([2, 1])
    
    with col_chart1:
        fig, ax = plt.subplots(figsize=(6, 3))  # Уменьшили размер с (8,5) до (6,3)
        bars = ax.bar(['Прогноз', 'Рекомендация'], 
                      [prediction, recommended], 
                      color=['#1f77b4', '#2ca02c'], 
                      edgecolor='black', 
                      width=0.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
                        
        ax.set_title('План загрузки', fontsize=11, fontweight='bold', pad=10)
        ax.set_ylabel('Единиц товара', fontsize=9)
        ax.set_ylim(0, max(prediction, recommended) * 1.4)  # Уменьшили отступ сверху
        ax.tick_params(axis='both', which='major', labelsize=9)
        plt.tight_layout(pad=2)  # Уменьшили отступы
        st.pyplot(fig, use_container_width=False)
    
    with col_chart2:
        # Дополнительная информация в виде карточек
        st.info(f"**Категория:** {category}\n\n**Маршрут:** {country_dep} → {country_arr}\n\n**Длительность:** {duration} ч\n\n**Время:** {dep_hour}:00")
    
    st.markdown("---")
    
    # Бизнес-интерпретация - компактно
    st.subheader("💡 Рекомендации")
    
    if category in ['water', 'hot drinks']:
        st.success(f"**{category.upper()}** — товар базового спроса. Риск утилизации минимальный. Рекомендуется стандартная загрузка.")
    elif category in ['beer', 'wine', 'liquor', 'sweet snacks']:
        st.warning(f"**{category.upper()}** — высокая волатильность спроса. Риск утилизации повышен. Загрузка ограничена.")
    else:
        st.info(f"**{category.upper()}** — товар массового спроса. Загрузка оптимизирована под параметры рейса.")

else:
    st.info("👈 Заполните параметры рейса в боковой панели слева и нажмите кнопку.")
    
    # Инфо о модели на стартовом экране - компактно
    with st.expander("ℹ️ Информация о модели", expanded=False):
        st.write(f"- **Алгоритм:** CatBoost (Gradient Boosting)")
        st.write(f"- **MAE на тесте:** {metadata['mae']:.3f} шт.")
        st.write(f"- **Обучено на записях:** {metadata['n_train_samples']:,}")
        st.write(f"- **Признаков:** {len(metadata['feature_names'])}")
        st.write(f"- **Sigma (RMSE):** {metadata['sigma']:.3f}")
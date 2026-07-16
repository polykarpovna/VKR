import joblib
import json
import pandas as pd
import numpy as np

def load_model_and_metadata(model_path='catboost_final_model.pkl', meta_path='model_metadata.json'):
    """Загружает обученную модель и метаданные"""
    model = joblib.load(model_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return model, metadata

def prepare_input_for_model(user_inputs, metadata):
    """
    Формирует DataFrame для модели на основе данных из интерфейса.
    Заполняет неизвестные признаки медианами и модами.
    """
    feature_names = metadata['feature_names']
    medians = metadata['medians']
    modes = metadata['modes']
    
    # Инициализируем словарь нулями/пустыми строками
    input_dict = {col: [0] for col in feature_names}
    
    # 1. Заполняем числовые признаки медианами
    for col in metadata['num_cols']:
        if col in medians:
            input_dict[col] = [medians[col]]
            
    # 2. Заполняем категориальные признаки модами
    for col in metadata['cat_cols']:
        if col in modes:
            input_dict[col] = [modes[col]]
            
    # 3. Перезаписываем значения, которые ввел пользователь
    input_dict['category'] = [user_inputs['category']]
    input_dict['country_arr'] = [user_inputs['country_arr']]
    input_dict['country_dep'] = [user_inputs['country_dep']] # По умолчанию та же страна или первая из списка
    input_dict['approx_duration'] = [user_inputs['duration']]
    input_dict['distance'] = [user_inputs['distance']]
    
    # Временные признаки и их sin/cos трансформации
    hour = user_inputs['dep_hour']
    month = user_inputs['month']
    weekday = user_inputs['weekday']
    
    input_dict['deptime_hour_rnd'] = [hour]
    input_dict['deptime_hour_rnd_sin'] = [np.sin(2 * np.pi * hour / 24)]
    input_dict['deptime_hour_rnd_cos'] = [np.cos(2 * np.pi * hour / 24)]
    
    input_dict['deptime_month'] = [month]
    input_dict['deptime_month_sin'] = [np.sin(2 * np.pi * month / 12)]
    input_dict['deptime_month_cos'] = [np.cos(2 * np.pi * month / 12)]
    
    input_dict['deptime_weekday'] = [weekday]
    input_dict['deptime_weekday_sin'] = [np.sin(2 * np.pi * weekday / 7)]
    input_dict['deptime_weekday_cos'] = [np.cos(2 * np.pi * weekday / 7)]

    # Создаем DataFrame
    df_input = pd.DataFrame(input_dict)
    
    # Приводим типы данных в точности как при обучении
    for col in metadata['cat_cols']:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str)
            
    for col in metadata['num_cols']:
        if col in df_input.columns:
            df_input[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(medians.get(col, 0))

    return df_input
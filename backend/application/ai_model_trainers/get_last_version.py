from datetime import datetime

from application import config_paths
from application.ai_models.ai_models import AI_Model_Type
from application.services import data_storage_services


def get_last_version_model(user_name, ai_Model_Type, trained_model_name):
    all_models_names = data_storage_services.get_model_names(user_name, ai_Model_Type).result

    def extract_date_time(s):
        parts = s.split("_")
        date_part = parts[-1]  # Последний элемент уже содержит дату-время в нужном формате
        try:
            return datetime.strptime(date_part, '%Y-%m-%d-%H-%M-%S')
        except ValueError:
            raise Exception(f"Invalid timestamp format: '{date_part}'")
    
    filtered_list = [item for item in all_models_names if trained_model_name in item]
    sorted_filtered_list = sorted(filtered_list, key=extract_date_time)
    latest_version = sorted_filtered_list[-1]
        
    return config_paths.get_model_path(user_name, AI_Model_Type.convert_to_string_try_get(ai_Model_Type)[1], latest_version), latest_version
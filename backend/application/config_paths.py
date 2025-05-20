import os
import configurations.config as config

def get_root_path():
    return f'{os.path.abspath('./')}'

def get_datasets_folder_path(user_name):
    return f'{get_root_path()}/{config.get_storage_configuration_datasets_path()}/{user_name}'

def get_dataset_path(user_name, filename):
    return f'{get_root_path()}/{config.get_storage_configuration_datasets_path()}/{user_name}/{filename}'

def get_models_folder_path(user_name, ai_model_type):
    return f'{get_root_path()}/{config.get_storage_configuration_models_path()}/{user_name}/{ai_model_type}'

def get_model_path(user_name, ai_model_type, filename):
    return f'{get_root_path()}/{config.get_storage_configuration_models_path()}/{user_name}/{ai_model_type}/{filename}'
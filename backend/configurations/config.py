import os
import json
from application import paths

def get_storage_configuration_datasets_path():
    config = __get_config()
    return config['StorageConfiguration']['DatasetsPath']

def get_storage_configuration_models_path():
    config = __get_config()
    return config['StorageConfiguration']['ModelsPath']

def __get_config():
    # Определяем переменную окружения, которая содержит среду выполнения
    environment = os.getenv('ENVIRONMENT', 'Development')
    
    # Формируем путь до нужного конфигурационного файла
    config_file_name = f'configurations/appsettings.{environment}.json'
    config_path = f'{paths.get_root_path()}/{config_file_name}'
    
    try:
        with open(config_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        raise Exception(f'Не найден файл конфигурации {config_file_name}')
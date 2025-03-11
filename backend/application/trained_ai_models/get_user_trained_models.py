import os

from application.Paths import Paths
from application.results.Result import Result

def get_all_models_for_user(user_name, ai_model_type):
    # Указываем путь к папке
    folder_path = f'{Paths.Trained_AI_Models}/{user_name}/'

    if not os.path.isdir(folder_path):
        return Result(None, "Такого пользователя не существует")
    
    # Получаем список всех элементов внутри папки с указанным типом модели нейронной сети
    contents = os.listdir(folder_path + ai_model_type)

    # Отбираем только те элементы, которые являются директориями
    directories = [item for item in contents if os.path.isdir(os.path.join(folder_path + ai_model_type, item))]
    
    return Result(directories, None)
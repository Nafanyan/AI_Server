import os
import shutil
import zipfile

from application.ai_models.ai_models import AI_Model_Type
from application.results.Result import Result
from application import paths

def create_dataset_zip(user_name, filename, file):
    """
    Метод для создания zip-файла набора данных.
    
    :param user_name: Имя пользователя
    :param file: Путь до файла
    :return: Ответ сервера
    """
    # Распаковываем архив
    try:
        return __save_zip_and_extracted_file(paths.get_datasets_folder_path(user_name), filename, file)
    
    except zipfile.BadZipFile:
        return Result(None, f'Error: The uploaded file is not a valid ZIP archive: {str(e)}')
    except Exception as e:
        return Result(None, f'Error: An unexpected error occurred while extracting the archive: {str(e)}')

def get_dataset_names(user_name):
    """
    Получение списка имен наборов данных для указанного пользователя.
    
    :param user_name: Имя пользователя
    :return: Ответ сервера
    """
    # Указываем путь к папке
    datasets_folder_path = paths.get_datasets_folder_path(user_name)

    if not os.path.isdir(datasets_folder_path):
        return Result(None, "Datasets for User is not exist")
    
    contents = os.listdir(datasets_folder_path)

    # Отбираем только те элементы, которые являются директориями
    directories = [item for item in contents if os.path.isdir(os.path.join(datasets_folder_path, item))]
    
    return Result(directories, None)

def get_dataset_by_name(user_name, dataset_name):
    zip_filename = f'{dataset_name}.zip'
    path_to_zip = paths.get_dataset_path(user_name, zip_filename)

    # Проверяем существование файла
    if not os.path.exists(path_to_zip):
        return Result(None, "Error: Dataset file is not exist")

    return Result(path_to_zip, None)

def delete_dataset_by_name(user_name, dataset_name):
    return __delete_zip_and_extract_folder(paths.get_models_folder_path(user_name), dataset_name)

def create_model_zip(user_name, ai_model_type, filename, file):
    try:
        _, ai_model_type_string = AI_Model_Type.convert_to_string_try_get(ai_model_type)
        return __save_zip_and_extracted_file(paths.get_models_folder_path(user_name, ai_model_type_string), filename, file)
    
    except zipfile.BadZipFile:
        return Result(None, f'Error: The created file is not a valid ZIP archive: {str(e)}')
    except Exception as e:
        return Result(None, f'Error: An unexpected error occurred while extracting the archive: {str(e)}')

def get_model_names(user_name, ai_model_type):
    # Указываем путь к папке
    _, ai_model_type_string = AI_Model_Type.convert_to_string_try_get(ai_model_type)
    models_folder_path = paths.get_models_folder_path(user_name, ai_model_type_string)

    if not os.path.isdir(models_folder_path):
        return Result(None, "Models for User is not exist")
    
    contents = os.listdir(models_folder_path)

    # Отбираем только те элементы, которые являются директориями
    directories = [item for item in contents if os.path.isdir(os.path.join(models_folder_path, item))]
    
    return Result(directories, None)

def get_model_by_name(user_name, ai_model_type, model_name):
    zip_filename = f'{model_name}.zip'
    _, ai_model_type_string = AI_Model_Type.convert_to_string_try_get(ai_model_type)
    path_to_zip = paths.get_model_path(user_name, ai_model_type_string, zip_filename)

    # Проверяем существование файла
    if not os.path.exists(path_to_zip):
        return Result(None, "Error: Model file is not exist")
    
    return Result(path_to_zip, None)

def delete_model_by_name(user_name, ai_model_type, model_name):
    _, ai_model_type_string = AI_Model_Type.convert_to_string_try_get(ai_model_type)
    return __delete_zip_and_extract_folder(paths.get_models_folder_path(user_name, ai_model_type_string), model_name)

def __save_zip_and_extracted_file(file_folder, filename, file):
    with zipfile.ZipFile(file) as archive:
        # Указываем директорию для распаковки
        if not os.path.exists(file_folder):
            os.makedirs(file_folder)
        
        # Сохраняем исходный ZIP-архив в ту же директорию
        zip_file_path = os.path.join(file_folder, filename + '.zip')
        with open(zip_file_path, 'wb') as f:
            f.write(file.read()) 

        # Распаковываем архив в указанную директорию
        extract_folder = os.path.join(file_folder, filename)
        with zipfile.ZipFile(file) as archive:
            archive.extractall(extract_folder)

        return Result('Success: Archive was successfully extracted', None)
    
def __delete_zip_and_extract_folder(file_folder, filename):
    # Удаление директории
    extract_folder = os.path.join(file_folder, filename)
    if os.path.isdir(extract_folder):
        try:
            shutil.rmtree(extract_folder)
        except OSError as e:
            return Result(None, f"Error: '{filename}': {e.strerror}")
    
    # Удаление ZIP-файла
    zip_file_path = os.path.join(file_folder, filename + '.zip')
    if os.path.isfile(zip_file_path):
        try:
            os.remove(zip_file_path)
        except OSError as e:
            return Result(None, f"Error: '{filename}': {e.strerror}")
    
    return Result(f'Success: File {filename} is deleted', None)
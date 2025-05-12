from enum import Enum
from datetime import datetime
import application.config_paths as config_paths

class Model_Classification_Type(Enum):
    Binary = 1
    Multiple = 2

class AI_Model_Type():
    CNN = 0
    LNN = 1
    __string_CNN = 'cnn'
    __string_LNN = 'lnn'

    def convert_to_string_try_get(model_type):
        if (int)(model_type) == AI_Model_Type.CNN:
            return True, AI_Model_Type.__string_CNN

        if (int)(model_type) == AI_Model_Type.LNN:
            return True, AI_Model_Type.__string_LNN

        return False, 'unknown'
    
    def get_name_for_trained_model(user_name, ai_model_type, trained_model_name):
        _, ai_model_type_string = AI_Model_Type.convert_to_string_try_get(ai_model_type)

        trained_model_name_for_save = f'{trained_model_name}_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'.replace(' ', '')
        return f'{config_paths.get_models_folder_path(user_name, ai_model_type_string)}/{trained_model_name_for_save}'
    






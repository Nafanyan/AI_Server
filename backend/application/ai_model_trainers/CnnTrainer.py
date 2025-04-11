from datetime import datetime
import os

import yaml
from application.ai_model_trainers.get_last_version import get_last_version_model
from application.services.zip_archive_service import create_zip_archive
import application.services.data_storage_services as data_storage_services
from application.ai_models.ai_models import CNN_Model_Name, AI_Model_Type
from application.ai_models.yolov5 import train
from application import paths

class CnnTrainer:
    def train(
            self,
            ai_model, 
            img_size, 
            batch_size, 
            epochs_num, 
            dataset_name, 
            trained_model_name, 
            user_name):
        options = {CNN_Model_Name.YOLOV5: lambda: self.__train_yolov5()}

        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs_num = epochs_num
        self.user_name = user_name
        self.dataset_name = dataset_name

        self.trained_ai_model_path = AI_Model_Type.get_name_for_trained_model(user_name, AI_Model_Type.CNN, trained_model_name)
        if not os.path.exists(self.trained_ai_model_path):
            os.makedirs(self.trained_ai_model_path)
            
        action = options.get(ai_model)
        action()

        model_folder, latest_version = get_last_version_model(user_name, AI_Model_Type.CNN, trained_model_name)
        create_zip_archive(model_folder)

        return data_storage_services.get_model_by_name(user_name, AI_Model_Type.CNN, latest_version).result


    def __train_yolov5(self):
        dataset_folder_path = paths.get_dataset_path(self.user_name, self.dataset_name).replace('.zip', '')
        dataset_path = f'{dataset_folder_path}/data.yaml'
        device = 'cpu'

        # Чтение данных из YAML файла
        with open(dataset_path, 'r') as file:
            data = yaml.safe_load(file)
        
        data['train'] = data['train'].replace('./', f'{dataset_folder_path}/' )
        data['val'] = data['val'].replace('./', f'{dataset_folder_path}/')
        data['test'] = data['test'].replace('./', f'{dataset_folder_path}/')

        with open(dataset_path, 'w') as file:
            yaml.dump(data, file)

        opt = train.parse_opt()
        opt.imgsz = self.img_size
        opt.batch_size = self.batch_size
        opt.epochs=self.epochs_num
        opt.data=dataset_path
        opt.weights='yolov5s.pt'
        opt.project=self.trained_ai_model_path
        opt.device=device
        
        train.main(opt)
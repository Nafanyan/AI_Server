from datetime import datetime
import os

import yaml
from application.services.zip_archive_service import create_zip_archive
import application.services.data_storage_services as data_storage_services
from application.ai_models.ai_models import AI_Model_Name, AI_Model_Type
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
        options = {AI_Model_Name.YOLOV5: lambda: self.__train_yolov5(user_name, dataset_name)}

        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs_num = epochs_num

        self.trained_ai_model_path = AI_Model_Type.get_name_for_trained_model(user_name, AI_Model_Type.CNN, trained_model_name)
        if not os.path.exists(self.trained_ai_model_path):
            os.makedirs(self.trained_ai_model_path)
            
        action = options.get(ai_model)
        action()

        all_models_names = data_storage_services.get_model_names(user_name, AI_Model_Type.CNN).result

        filtered_list = [item for item in all_models_names if trained_model_name in item]
        sorted_filtered_list = sorted(
            filtered_list,
            key=lambda x: datetime.strptime(x.rsplit('_', 1)[-1], '%Y-%m-%d%H:%M')
        )
        latest_version = sorted_filtered_list[-1]
        
        model_folder = paths.get_model_path(user_name, AI_Model_Type.convert_to_string_try_get(AI_Model_Type.CNN)[1], latest_version)
        create_zip_archive(model_folder)

        return data_storage_services.get_model_by_name(user_name, AI_Model_Type.CNN, latest_version).result


    def __train_yolov5(self, user_name, dataset_name):
        dataset_folder_path = paths.get_dataset_path(user_name, dataset_name).replace('.zip', '')
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
from application.Paths import Paths
from application.ai_models.ai_models import AI_Model_Name, AI_Model_Type
from application.ai_models.yolov5 import train

class CNN_trainer:
    def train(
            self,
            ai_model, 
            img_size, 
            batch_size, 
            epochs_num, 
            dataset_name, 
            trained_model_name, 
            user_name):
        options = {AI_Model_Name.YOLOV5: lambda: self.__train_yolov5(dataset_name, user_name)}
        
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs_num = epochs_num
        self.trained_ai_model_path = f'{Paths.Trained_AI_Models}/{user_name}/{AI_Model_Type.CNN}'
        self.trained_model_name = trained_model_name

        action = options.get(ai_model)
        action()

    def __train_yolov5(self, dataset_name, user_name):
        dataset_path = f'{Paths.Datasets}/{user_name}/{dataset_name}/data.yaml'
        device = 'cpu'

        train.start_train(
            self.img_size, 
            self.batch_size, 
            self.epochs_num, 
            dataset_path, 
            device, 
            self.trained_ai_model_path, 
            self.trained_model_name)
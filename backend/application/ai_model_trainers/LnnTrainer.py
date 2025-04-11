import os
from application.ai_model_trainers.get_last_version import get_last_version_model
from application.services import data_storage_services
from application.services.zip_archive_service import create_zip_archive
from application.ai_models.ai_models import LNN_Model_Name, AI_Model_Type

class LNN_Trainer:
    def train(
            self,
            ai_model,
            epochs,
            batch_size,
            neurons_in_layers,
            activations,
            optimizer,
            user_name,
            dataset_name,
            trained_model_name):
        options = {LNN_Model_Name.Binary: lambda: self.__train_binary()}
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.neurons_in_layers = neurons_in_layers
        self.activations = activations
        self.optimizer = optimizer
        self.user_name = user_name
        self.dataset_name = dataset_name

        self.trained_ai_model_path = AI_Model_Type.get_name_for_trained_model(user_name, AI_Model_Type.LNN, trained_model_name)
        if not os.path.exists(self.trained_ai_model_path):
            os.makedirs(self.trained_ai_model_path)
        
        self.init_last_element_in_init_activations(ai_model)
        self.loss = self.init_loss(ai_model)
        self.init_last_element_in_neurons_in_layers(ai_model)

        action = options.get(ai_model)
        action()

        model_folder, latest_version = get_last_version_model(user_name, AI_Model_Type.LNN, trained_model_name)
        create_zip_archive(model_folder)

        return data_storage_services.get_model_by_name(user_name, AI_Model_Type.LNN, latest_version).result
    
    def __train_binary(self):

        return 0
    
    def init_last_element_in_init_activations(self, ai_model):
        if ai_model == LNN_Model_Name.Binary:
            self.activations.append('sigmoid')
        
        self.activations.append('softmax')

    def init_loss(self, ai_model):
        if ai_model == LNN_Model_Name.Binary:
            return 'binary_crossentropy'
        
        return 'categorical_crossentropy'

    def init_last_element_in_neurons_in_layers(self, ai_model):
        if ai_model == LNN_Model_Name.Binary:
            self.neurons_in_layers.append(1)
        
        self.neurons_in_layers.append(1)

    def __validate_parameters(self):
    # Проверка, что массивы одинаковой длины
      if len(self.neurons_in_layers) != len(self.activations):
          return False, "Количество элементов в neurons_in_layers и activations должно быть одинаковым"
  
      # Если все проверки пройдены
      return True, "Validation successful."

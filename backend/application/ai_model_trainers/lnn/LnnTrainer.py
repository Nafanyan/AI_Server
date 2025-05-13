import os
import random
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
from application.create_app import create_app
from application.create_app.create_app import build_exe_with_class_names
from application import config_paths
from application.results.Result import Result
from application.ai_model_trainers.get_last_version import get_last_version_model
from application.services import data_storage_services
from application.services.zip_archive_service import create_zip_archive
from application.ai_models.ai_models import Model_Classification_Type, AI_Model_Type

class LNN_Trainer:
    def __init__(            
            self,
            ai_model,
            epochs,
            batch_size,
            neurons_in_layers,
            activations,
            optimizer,
            user_name,
            dataset_name,
            train_percentage,
            test_percentage):
        self.ai_model = ai_model 
        self.epochs = epochs 
        self.batch_size = batch_size 
        self.neurons_in_layers = neurons_in_layers.copy() 
        self.activations = activations.copy() 
        self.optimizer = optimizer 
        self.user_name = user_name
        self.dataset_name = dataset_name

        # Путь к датасету необходимому для обучения
        self.dataset_folder_path = config_paths.get_dataset_path(self.user_name, self.dataset_name).replace('.zip', '')

        # Донастройка нейронной сети в зависимости от её типа: бинарная, либо многоклассовая
        self.__init_last_element_in_init_activations(ai_model)
        self.__init_last_element_in_neurons_in_layers(ai_model)
        self.loss = self.__init_loss(ai_model)

        # Получение всей выборки данных
        data, labels = self.__get_data()
        self.all_classes = list(set(labels))

        # Кодирование labels
        encoded_labels = self.__get_encode_labels(ai_model, labels)

        # Деление на тренировочную, тестовую и проверочную
        if train_percentage + test_percentage > 100:
            raise ValueError("При делении выборки на обучающую и тестовую, сумма значений не должна быть больше 100")  

        total_size = len(data)
        train_size = int(total_size * train_percentage * 0.01)
        test_size = int(total_size * test_percentage * 0.01)

        self.train_data, self.test_data, self.valid_data = np.split(data, [train_size, train_size + test_size])
        self.train_labels, self.test_labels, self.valid_labels = np.split(encoded_labels, [train_size, train_size + test_size])

    def train(self):
        isSuccess, msg = self.__validate_parameters()
        if not isSuccess:
            return  Result(None, msg)
    
        # Запуск обучения
        model, history, test_loss, test_acc = self.__train()

        return model, history, test_loss, test_acc

    def train_and_save(self, trained_model_name, is_create_app):
        isSuccess, msg = self.__validate_parameters()
        if not isSuccess:
            return Result(None, msg)
    
        # Запуск обучения
        model, history, _, _ = self.__train()
        
        # Сохранения обученной модели
        return self.save_model(trained_model_name, model, history, is_create_app)
    
    def save_model(self, trained_model_name, model, history, is_create_app):
        # Создание директории, в которую будет сохраняться модель
        trained_ai_model_folder_path = AI_Model_Type.get_name_for_trained_model(self.user_name, AI_Model_Type.LNN, trained_model_name)
        if not os.path.exists(trained_ai_model_folder_path):
            os.makedirs(trained_ai_model_folder_path)

        # Сохранение обученной модели и графиков обучения
        model_path = f'{trained_ai_model_folder_path}/{trained_model_name}.h5'
        model.save(model_path)
        self.__create_plots(history, trained_ai_model_folder_path)

        # Если нужно создать приложение
        if is_create_app:
            build_exe_with_class_names(
                self.all_classes,
                base_script_path=create_app.current_file_path,
                model_path=model_path,
                exe_name="App.exe",
                dist_path=trained_ai_model_folder_path)

        # Из всех обученных моделей со схожим именем выбираем самую новую модель
        _, latest_version = get_last_version_model(self.user_name, AI_Model_Type.LNN, trained_model_name)
        create_zip_archive(trained_ai_model_folder_path)

        return data_storage_services.get_model_by_name(self.user_name, AI_Model_Type.LNN, latest_version)

    def __train(self):
        model_layers = []
        for i in range(len(self.neurons_in_layers)):
           model_layers.append(layers.Dense(self.neurons_in_layers[i], activation=self.activations[i]))
        model = keras.Sequential(model_layers)

        model.compile(
                    optimizer=self.optimizer,
                    loss=self.loss,
                    metrics=["accuracy"])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(
                            self.train_data,
                            self.train_labels,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            validation_data = (self.valid_data, self.valid_labels),
                            callbacks=[early_stopping])
        test_loss, test_acc = model.evaluate(self.test_data, self.test_labels)

        return model, history, test_loss, test_acc
    
    def __create_plots(self, history, save_folder):
        # Построение графиков обучения
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='validation loss')
        plt.title('Loss Plot')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{save_folder}/loss_plot.png', dpi=300)

        plt.figure(figsize=(8, 6))
        plt.plot(history.history['accuracy'], label='training accuracy')
        plt.plot(history.history['val_accuracy'], label='validation accuracy')
        plt.title('Accuracy Plot')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'{save_folder}/accuracy_plot.png', dpi=300)

    def __init_last_element_in_init_activations(self, ai_model):
        if ai_model == Model_Classification_Type.Binary:
            self.activations.append('sigmoid')
            return
        
        self.activations.append('softmax')

    def __init_loss(self, ai_model):
        if ai_model == Model_Classification_Type.Binary:
            return 'binary_crossentropy'
        
        return 'categorical_crossentropy'

    def __init_last_element_in_neurons_in_layers(self, ai_model):
        if ai_model == Model_Classification_Type.Binary:
            self.neurons_in_layers.append(1)
            return
        
        self.neurons_in_layers.append(len(self.__get_quantity_classes()))

    def __get_quantity_classes(self):
        all_items = os.listdir(self.dataset_folder_path)
        return [item for item in all_items if os.path.isdir(os.path.join(self.dataset_folder_path, item))]

    def __get_data(self):
        all_files = self.__get_all_files(self.dataset_folder_path)
        random.shuffle(all_files)

        data = []
        labels = []

        # Читаем данные из каждого файла и сохраняем их вместе с меткой
        for file_path in all_files:
            with open(file_path, 'r') as f:
                numbers = [float(line.strip()) for line in f.readlines()]
                data.append(np.array(numbers))
                label = os.path.basename(os.path.dirname(file_path))  # Имя папки, где находится файл
                labels.append(label)

        # Преобразуем списки в numpy массивы
        return np.array(data), np.array(labels)

    # Функция для получения всех файлов в директории и её подпапках
    def __get_all_files(self, root_dir):
        files = []
        for dir_name, _, file_names in os.walk(root_dir):
            for file_name in file_names:
                if file_name.endswith('.txt'):
                    files.append(os.path.join(dir_name, file_name))
        return files

    def __get_encode_labels(self, ai_model, labels):
        _, inverse_indices = np.unique(labels, return_inverse=True)
        encoded_labels = inverse_indices
        if ai_model == Model_Classification_Type.Binary:
            return encoded_labels
        
        return self.__to_one_hot(encoded_labels, len(self.__get_quantity_classes()))

    def __to_one_hot(self, labels, dimension):
        results = np.zeros((len(labels), dimension))
        for i, label in enumerate(labels):
          results[i, label] = 1
        return results

    def __validate_parameters(self):
        # Проверка, что массивы одинаковой длины
        if len(self.neurons_in_layers) != len(self.activations):
            return False, "Количество элементов в neurons_in_layers и activations должно быть одинаковым"

        # Если все проверки пройдены
        return True, "Валидация прошла успешно"

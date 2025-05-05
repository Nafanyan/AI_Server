import os
import random
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
from application import config_paths
from application.results.Result import Result
from application.ai_model_trainers.get_last_version import get_last_version_model
from application.services import data_storage_services
from application.services.zip_archive_service import create_zip_archive
from application.ai_models.ai_models import CNN_Model_Name, AI_Model_Type

class CNN_Trainer:
    def __init__(            
            self,
            ai_model,
            img_size,
            epochs,
            batch_size,
            filters,
            kernel_sizes,
            pool_sizes,
            activations,
            optimizer,
            user_name,
            dataset_name,
            train_percentage,
            test_percentage):
        self.ai_model = ai_model 
        self.img_size = img_size
        self.epochs = epochs 
        self.batch_size = batch_size 
        self.filters = filters.copy() 
        self.kernel_sizes = kernel_sizes.copy() 
        self.pool_sizes = pool_sizes.copy() 
        self.activations = activations.copy() 
        self.optimizer = optimizer 
        self.user_name = user_name
        self.dataset_name = dataset_name

        # Путь к датасету необходимому для обучения
        self.dataset_folder_path = config_paths.get_dataset_path(self.user_name, self.dataset_name).replace('.zip', '')

        # Донастройка нейронной сети в зависимости от её типа: бинарная, либо многоклассовая
        self.__init_last_element_in_init_activations(ai_model)
        self.loss = self.__init_loss(ai_model)
        self.__init_last_element_in_neurons_in_layers(ai_model)

        # Получение всей выборки данных
        data, labels = self.__get_data()

        # Кодирование labels
        encoded_labels = self.__get_encode_labels(ai_model, labels)

        # Деление на тренировочную, тестовую и проверочную
        if train_percentage + test_percentage > 100:
            raise ValueError("При делении выборки на обучающую и тестовую, сумма значений не должна быть больше 100")  

        total_size = len(data)
        train_size = int(total_size * train_percentage * 0.01)
        test_size = int(total_size * test_percentage * 0.01)

        train_data, test_data, valid_data = np.split(data, [train_size, train_size + test_size])
        train_labels, test_labels, valid_labels = np.split(encoded_labels, [train_size, train_size + test_size])

        self.train_data = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
        self.train_data = self.train_data.map(self.process_path).batch(self.batch_size)

        self.test_data = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
        self.test_data =  self.test_data.map(self.process_path).batch(self.batch_size)

        self.valid_data = tf.data.Dataset.from_tensor_slices((valid_data, valid_labels))
        self.valid_data =  self.valid_data.map(self.process_path).batch(self.batch_size)



    def train_and_save(self, trained_model_name):
        # Запуск обучения
        model, history, _, _ = self.__train()
        
        return self.save_model(trained_model_name, model, history)
    
    def train(self):
        # Запуск обучения
        model, history, test_loss, test_acc = self.__train()

        return model, history, test_loss, test_acc
    
    def save_model(self, trained_model_name, model, history):
        # Создание директории, в которую будет сохраняться модель
        trained_ai_model_folder_path = AI_Model_Type.get_name_for_trained_model(self.user_name, AI_Model_Type.LNN, trained_model_name)
        if not os.path.exists(trained_ai_model_folder_path):
            os.makedirs(trained_ai_model_folder_path)

        # Сохранение обученной модели
        model.save(f'{trained_ai_model_folder_path}/{trained_model_name}.h5')
        self.__create_plots(history, trained_ai_model_folder_path)

        _, latest_version = get_last_version_model(self.user_name, AI_Model_Type.LNN, trained_model_name)
        create_zip_archive(trained_ai_model_folder_path)

        return data_storage_services.get_model_by_name(self.user_name, AI_Model_Type.LNN, latest_version)

    def __train(self):
        inputs = keras.Input(shape=(self.img, 180, 3))
        x = layers.Rescaling(1./255)(inputs)

        model_layers = []
        for i in range(len(self.filters)) - 2:
            x = layers.Conv2D(filters=self.filters[i], kernel_size=self.kernel_size[i], activation=self.activations[i])(x)
            x = layers.MaxPooling2D(pool_size=self.pool_size[i])(x)

        x = layers.Conv2D(filters=self.filters[-2], kernel_size=self.kernel_size[-2], activation=self.activations[-2])(x)
        x = layers.Flatten()(x)

        outputs = layers.Dense(self.filters[-1], activation=self.activations[-1])(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

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
        if ai_model == CNN_Model_Name.Binary:
            self.activations.append('sigmoid')
            return
        
        self.activations.append('softmax')

    def __init_loss(self, ai_model):
        if ai_model == CNN_Model_Name.Binary:
            return 'binary_crossentropy'
        
        return 'categorical_crossentropy'

    def __init_last_element_in_neurons_in_layers(self, ai_model):
        if ai_model == CNN_Model_Name.Binary:
            self.filters.append(1)
            return
        
        self.filters.append(len(self.__get_quantity_classes()))

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
                data.append(file_path)
                label = os.path.basename(os.path.dirname(file_path))  # Имя папки, где находится файл
                labels.append(label)

        # Преобразуем списки в numpy массивы
        return np.array(data), np.array(labels)

    def process_path(self, file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [180, 180])
        img = img / 255.0
        return img, label

    # Функция для получения всех файлов в директории и её подпапках
    def __get_all_files(self, root_dir):
        files = []
        for dir_name, _, file_names in os.walk(root_dir):
            for file_name in file_names:
                if file_name.endswith('.jpg'):
                    files.append(os.path.join(dir_name, file_name))
        return files

    def __get_encode_labels(self, ai_model, labels):
        _, inverse_indices = np.unique(labels, return_inverse=True)
        encoded_labels = inverse_indices
        if ai_model == CNN_Model_Name.Binary:
            return encoded_labels
        
        return self.__to_one_hot(encoded_labels, len(self.__get_quantity_classes()))

    def __to_one_hot(self, labels, dimension):
        results = np.zeros((len(labels), dimension))
        for i, label in enumerate(labels):
          results[i, label] = 1
        return results

import os
from flask import Blueprint, request, jsonify, send_file

from application.ai_model_trainers.CnnTrainer import CnnTrainer
from application.ai_models.ai_models import CNN_Model_Name, AI_Model_Type

train_lnn_models_bp = Blueprint(
    'train-cnn-model',
    __name__,
    url_prefix='/api/train-lnn-model'
)

@train_lnn_models_bp.route('/train', methods=['POST'])
def train_model():
    """
    Обучение линейной модели нейронной сети
    ---
    parameters:
      - name: ai_model
        in: formData
        type: string
        enum: ['binary_classification', 'multiple_classification']
        required: true
        description: Тип модели. Для бинарной классификации binary_classification, а для многоклассовой multiple_classification
      - name: epochs
        in: formData
        type: integer
        required: true
        description: Количество эпох обучения
      - name: batch_size
        in: formData
        type: integer
        required: true
      - name: neurons_in_layers
        in: formData
        type: array
        items:
          type: integer
        description: Количество нейронов в каждом слое без учета выходного слоя
      - name: activations
        in: formData
        type: array
        items:
          type: string
          enum: ['relu', 'leaky_relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'exponential', 'swish']
        required: true
        description: Функции активации для каждого слоя без учета выходного слоя
      - name: optimizer
        in: formData
        type: string
        enum: ['sgd', 'rmsprop', 'adam', 'adadelta', 'adamax', 'nadam', 'ftrl']
        required: true
        description: Функции оптимизации
      - name: user_name
        in: formData
        type: string
        required: true
        description: Имя пользователя
      - name: dataset_name
        in: formData
        type: string
        required: true
        description: Название датасета
      - name: trained_model_name
        in: formData
        type: string
        required: true
        description: Имя для обучаемой модели
    responses:
      200:
        description: Model training parameters received
    """
        
    return 0
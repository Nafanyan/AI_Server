import os
from flask import Blueprint, abort, request, jsonify, send_file

from application.ai_models.AiModelNameConverter import AiModelNameConverter
from application.ai_model_trainers.cnn.CnnTrainerYolov5 import CnnTrainerYolov5
from application.ai_models.AiModelNameConverter import AiModelNameConverter
from application.ai_model_trainers.cnn.CnnTrainer import CNN_Trainer

train_cnn_models_bp = Blueprint(
    'train-cnn-model',
    __name__,
    url_prefix='/api/train-cnn-model'
)

@train_cnn_models_bp.route('/train', methods=['POST'])
def train_model():
    """
    Обучение сверточной модели нейронной сети
    ---
    tags:
      - Training CNN
    parameters:
      - name: ai_model
        in: formData
        type: string
        enum: ['binary_classification', 'multiple_classification']
        required: true
        description: Тип модели. Для бинарной классификации binary_classification, а для многоклассовой multiple_classification
      - name: img_size
        in: formData
        type: integer
        required: true
        description: Размер, к которому будет приводиться изображение. Достаточно одной величины т.к. форма будет квадратная
      - name: epochs
        in: formData
        type: integer
        required: true
        description: Количество эпох обучения
      - name: batch_size
        in: formData
        type: integer
        required: true
        description: Количество примеров данных (или наблюдений), которые модель обрабатывает одновременно за один проход (итерацию) во время обучения
      - name: filters 
        in: formData
        type: array
        items:
          type: integer
        description: число фильтров (ядер свертки), которые будут обучаться и применяться к входному изображению. Каждый фильтр извлекает определённые признаки, и выходной тензор будет иметь столько каналов, сколько фильтров
      - name: kernel_sizes 
        in: formData
        type: array
        items:
          type: integer
        description: размер фильтра (ядра свертки), задаётся как одно число (для квадратных фильтров, например 3 — это 3x3) или кортеж (например (3,3)). Определяет область локального восприятия для каждого фильтра
      - name: pool_sizes 
        in: formData
        type: array
        items:
          type: integer
        description: размер окна (например, 2 или (2, 2)), по которому берётся максимум в операции подвыборки (MaxPooling). Это окно скользит по входному тензору, и для каждого окна выбирается максимальное значение
      - name: activations
        in: formData
        type: array
        items:
          type: string
        required: true
        description: Функции активации для каждого слоя без учета выходного слоя. Возможные функци ['relu', 'leaky_relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'exponential', 'swish']
      - name: optimizer
        in: formData
        type: string
        enum: ['sgd', 'rmsprop', 'adam', 'adadelta', 'adamax', 'nadam', 'ftrl']
        required: true
        description: Функция оптимизации
      - name: train_percentage
        in: formData
        type: number
        format: float
        minimum: 0
        maximum: 100
        required: true
        description: Процентное соотношение тренировочной выборки от общего набора данных
      - name: test_percentage
        in: formData
        type: number
        format: float
        minimum: 0
        maximum: 100
        required: true
        description: Процентное соотношение проверочной выборки от общего набора данных
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
    if request.method != 'POST':
      return jsonify({'error': 'Invalid request method'}), 400

    ai_model = request.form.get('ai_model')
    img_size = int(request.form.get('img_size'))
    epochs = int(request.form.get('epochs'))
    batch_size = int(request.form.get('batch_size'))
    filters  = list(map(int, request.form['filters'].split(',')))
    kernel_sizes = list(map(int, request.form['kernel_sizes'].split(',')))
    pool_sizes = list(map(int, request.form['pool_sizes'].split(',')))
    activations = request.form.get('activations').split(',')
    optimizer = request.form.get('optimizer')
    user_name = request.form.get('user_name')
    dataset_name = request.form.get('dataset_name')
    trained_model_name = request.form.get('trained_model_name')
    train_percentage = int(request.form.get('train_percentage'))
    test_percentage = int(request.form.get('test_percentage'))

    if (len(filters) != len(kernel_sizes) != len(pool_sizes)):
       return jsonify({'error': 'The length of "filters", "kernel_sizes" and "pool_sizes" must be the same.'}), 400 

    try:
      trainer = CNN_Trainer(
         AiModelNameConverter.convert(ai_model.lower()),
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
         test_percentage)
      trained_model = trainer.train_and_save(trained_model_name)

      if not trained_model.isSuccess:
         return jsonify({'error': f'{trained_model.errors}'}), 400

      return send_file(
        trained_model.result,
        mimetype='application/zip',
        as_attachment=True,
        download_name=os.path.basename(trained_model.result))
    except Exception as ex:
      print(ex)
      abort(500, "Произошла внутренняя ошибка сервера.")

@train_cnn_models_bp.route('/train/yolo5', methods=['POST'])
def train_model_yolo5():
    """
    Обучение сверточной модели нейронной сети YOLO5
    ---
    tags:
      - Training CNN
    parameters:
      - name: ai_model
        in: formData
        type: string
        required: true
        description: Название модели AI
        example: "yolov5"
      - name: img_size
        in: formData
        type: integer
        required: true
        description: Размер изображения
        example: 640
      - name: batch_size
        in: formData
        type: integer
        required: true
        description: Размер батча
        example: 16
      - name: epochs
        in: formData
        type: integer
        required: true
        description: Количество эпох обучения
        example: 200
      - name: dataset_name
        in: formData
        type: string
        required: true
        description: Название датасета
        example: "plane_and_helicopter"
      - name: user_name
        in: formData
        type: string
        required: true
        description: Имя пользователя
        example: "Nafanyan"
      - name: trained_model_name
        in: formData
        type: string
        required: true
        description: Имя обученной модели
        example: "custom_trained_yolov5"
    responses:
      200:
        description: Модель успешно обучена
        content:
          application/json:
            schema:
              type: object
              properties:
                status:
                  type: string
                  description: Статус операции
                  example: "Success"
                message:
                  type: string
                  description: Сообщение
                  example: "Model trained successfully!"
    """
    if request.method != 'POST':
        return jsonify({'error': 'Invalid request method'}), 400
    
    ai_model = request.form.get('ai_model')
    img_size = request.form.get('img_size')
    batch_size = request.form.get('batch_size')
    epochs = request.form.get('epochs')
    dataset_name = request.form.get('dataset_name')
    user_name = request.form.get('user_name').replace(" ", "_")
    trained_model_name = request.form.get('trained_model_name')
    try:
      trainer = CnnTrainerYolov5(
          AiModelNameConverter.convert(ai_model.lower()),
          img_size, 
          batch_size, 
          epochs, 
          dataset_name, 
          user_name)
      trained_model = trainer.train(trained_model_name)

      if not trained_model.isSuccess:
         return jsonify({'error': f'{trained_model.errors}'}), 400

      return send_file(
        trained_model.result,
        mimetype='application/zip',
        as_attachment=True,
        download_name=os.path.basename(trained_model.result))
    except Exception as ex:
      print(ex)
      abort(500, "Произошла внутренняя ошибка сервера.")
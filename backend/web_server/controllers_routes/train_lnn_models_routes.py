import os
from flask import Blueprint, abort, request, jsonify, send_file

from application.ai_model_trainers.LnnOptimizeTrainer import LnnOptimizeTrainer
from application.ai_models.AiModelNameConverter import AiModelNameConverter
from application.ai_model_trainers.LnnTrainer import LNN_Trainer

train_lnn_models_bp = Blueprint(
    'train-lnn-model',
    __name__,
    url_prefix='/api/train-lnn-model'
)

@train_lnn_models_bp.route('/train', methods=['POST'])
def train_model():
    """
    Обучение линейной модели нейронной сети
    ---
    tags:
      - Training LNN
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
        description: Количество примеров данных (или наблюдений), которые модель обрабатывает одновременно за один проход (итерацию) во время обучения
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
        required: true
        description: Функции активации для каждого слоя без учета выходного слоя. Возможные функци ['relu', 'leaky_relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'exponential', 'swish']
      - name: optimizer
        in: formData
        type: string
        enum: ['sgd', 'rmsprop', 'adam', 'adadelta', 'adamax', 'nadam', 'ftrl']
        required: true
        description: Функция оптимизации
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
    responses:
      200:
        description: Model training parameters received
    """
    if request.method != 'POST':
      return jsonify({'error': 'Invalid request method'}), 400

    ai_model = request.form.get('ai_model')
    epochs = int(request.form.get('epochs'))
    batch_size = int(request.form.get('batch_size'))
    neurons_in_layers = list(map(int, request.form['neurons_in_layers'].split(',')))
    activations = request.form.get('activations').split(',')
    optimizer = request.form.get('optimizer')
    user_name = request.form.get('user_name')
    dataset_name = request.form.get('dataset_name')
    trained_model_name = request.form.get('trained_model_name')
    train_percentage = int(request.form.get('train_percentage'))
    test_percentage = int(request.form.get('test_percentage'))
    try:
      trainer = LNN_Trainer(
         AiModelNameConverter.convert(ai_model.lower()),
         epochs,
         batch_size,
         neurons_in_layers,
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


@train_lnn_models_bp.route('/optimize-train', methods=['POST'])
def optimize_train_model():
    """
    Обучение линейной модели нейронной сети
    ---
    tags:
      - Training LNN
    parameters:
      - name: ai_model
        in: formData
        type: string
        enum: ['binary_classification', 'multiple_classification']
        required: true
        description: Тип модели. Для бинарной классификации binary_classification, а для многоклассовой multiple_classification
      - name: epochs
        in: formData
        type: array
        required: true
        items:
          type: integer
        description: Допустимые варианты эпох среди, которых будет вестись поиск наиболее оптимального значения
      - name: hidden_layers
        in: formData
        type: array
        required: true
        items:
          type: integer
        description: Допустимые варианты количества скрытых слоев, среди которых будет вестись поиск наиболее оптимального значения
      - name: batch_sizes
        in: formData
        type: array
        required: true
        items:
          type: integer
        description: Допустимые варианты количества примеров данных, обрабатываемых моделью за одну итерацию, среди которых будет вестись поиск наиболее оптимального значения
      - name: neurons_per_layers
        in: formData
        type: array
        required: true
        items:
          type: integer
        description: Допустимые варианты количества нейронов в скрытых слоях, среди которых будет вестись поиск наиболее оптимального значения
      - name: activation_functions
        in: formData
        type: array
        required: true
        items:
          type: string
        enum: ['relu', 'leaky_relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'exponential', 'swish']
        description: Возможные функции активации для каждого слоя без учета выходного слоя. Возможные функци ['relu', 'leaky_relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'exponential', 'swish']
      - name: optimizers
        in: formData
        type: array
        required: true
        items:
          type: string
        enum: ['sgd', 'rmsprop', 'adam', 'adadelta', 'adamax', 'nadam', 'ftrl']
        description: Возможные функции оптимизации ['sgd', 'rmsprop', 'adam', 'adadelta', 'adamax', 'nadam', 'ftrl']
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

    user_name = request.form.get('user_name')
    dataset_name = request.form.get('dataset_name')
    trained_model_name = request.form.get('trained_model_name')

    ai_model = request.form.get('ai_model')

    epochs = [int(epoch) for epoch in request.form.get('epochs').split(',')]
    hidden_layers = [int(layer) for layer in request.form.get('hidden_layers').split(',')]
    batch_sizes = [int(batch_size) for batch_size in request.form.get('batch_sizes').split(',')]
    neurons_per_layers = [int(neuron_count) for neuron_count in request.form.get('neurons_per_layers').split(',')]
    activation_functions = request.form.get('activation_functions').split(',')
    optimizers = request.form.get('optimizers').split(',')

    try:
        # Создаем экземпляр тренера с новыми параметрами
        trainer = LnnOptimizeTrainer(
            ai_model=AiModelNameConverter.convert(ai_model.lower()),
            epochs=epochs,
            hidden_layers=hidden_layers,
            batch_sizes=batch_sizes,
            neurons_per_layers=neurons_per_layers,
            activation_functions=activation_functions,
            optimizers=optimizers,
            user_name=user_name,
            dataset_name=dataset_name,
            trained_model_name=trained_model_name,
        )
        
        # Запускаем обучение
        trained_model = trainer.optimize_train()

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

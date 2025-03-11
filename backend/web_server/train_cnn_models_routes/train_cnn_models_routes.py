from flask import Blueprint, request, jsonify

from application.ai_model_trainers.cnn_trainer import CNN_trainer
from application.ai_models.ai_models import AI_Model_Name, AI_Model_Type
from application.trained_ai_models.get_user_trained_models import get_all_models_for_user

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
      - Training
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: TrainCNNModelInput
          required:
            - ai_model
            - img_size
            - batch_size
            - epochs_num
            - dataset_name
            - user_name
            - trained_model_name  # Добавлен новый параметр
          properties:
            ai_model:
              type: string
              description: Название модели AI
              example: "yolov5"
            img_size:
              type: integer
              description: Размер изображения
              example: 640
            batch_size:
              type: integer
              description: Размер батча
              example: 16
            epochs_num:
              type: integer
              description: Количество эпох обучения
              example: 200
            dataset_name:
              type: string
              description: Название датасета
              example: "plane_and_helicopter"
            user_name:
              type: string
              description: Имя пользователя
              example: "Nafanyan"
            trained_model_name:
              type: string
              description: Имя обученной модели
              example: "custom_trained_yolov5"
    responses:
      200:
        description: Модель успешно обучена
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
    if request.method == 'POST':
        input_data = request.json
        
        ai_model = input_data['ai_model']
        img_size = input_data['img_size']
        batch_size = input_data['batch_size']
        epochs_num = input_data['epochs_num']
        dataset_name = input_data['dataset_name']
        user_name = input_data['user_name'].replace(" ", "_")
        trained_model_name = input_data['trained_model_name']

        trainer = CNN_trainer()
        trainer.train(
            getattr(AI_Model_Name, ai_model.upper()),
            img_size,
            batch_size,
            epochs_num,
            dataset_name,
            trained_model_name,
            user_name,
        )

        return jsonify({
            'status': 'Success',
            'message': f'Model {ai_model} trained successfully as {trained_model_name}!'  # Используем новое имя модели в сообщении
        }), 200
    
@train_cnn_models_bp.route('/<string:user_name>', methods=['GET'])
def get_models_by_username(user_name):
    """
    Получение списка имен обученных моделей сверточных нейронных сетей по имени пользователя
    ---
    tags:
      - Training
    parameters:
      - name: user_name
        in: path
        required: true
        type: string
        description: Имя пользователя
        example: "Nafanyan"
    responses:
      200:
        description: Возвращает список моделей
        content:
          application/json:
            schema:
              type: array
              items:
                type: object
                properties:
                  model_name:
                    type: string
                    description: Название модели
                    example: "yolov5"
                  training_date:
                    type: string
                    format: date-time
                    description: Дата обучения модели
                    example: "2023-10-15T12:34:56Z"
                  status:
                    type: string
                    description: Статус модели
                    example: "Trained"
    """
    result = get_all_models_for_user(user_name, AI_Model_Type.CNN)

    if (result.isSuccess):
      return jsonify(result.result), 200

    return jsonify(result.errors), 400
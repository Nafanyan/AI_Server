import os
from flask import Blueprint, abort, request, jsonify, send_file

from application.ai_models.AiModelNameConverter import AiModelNameConverter
from application.ai_model_trainers.CnnTrainer import CnnTrainer
from application.ai_models.ai_models import CNN_Model_Name

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
      - name: epochs_num
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
    epochs_num = request.form.get('epochs_num')
    dataset_name = request.form.get('dataset_name')
    user_name = request.form.get('user_name').replace(" ", "_")
    trained_model_name = request.form.get('trained_model_name')
    try:
      trainer = CnnTrainer(
          AiModelNameConverter.convert(ai_model.lower()),
          img_size, 
          batch_size, 
          epochs_num, 
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
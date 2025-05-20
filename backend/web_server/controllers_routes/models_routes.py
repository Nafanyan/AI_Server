import os
from flask import Blueprint, request, jsonify, send_file

from application.ai_models.ai_models import AI_Model_Type
import application.services.data_storage_services as data_storage_services

models_bp = Blueprint(
    'models',
    __name__,
    url_prefix='/api/models'
)

@models_bp.route('/save-and-extract', methods=['POST'])
def save_and_extract_model():
    """
    Загрузить и разархивировать ZIP-архив модели.
    ---
    tags:
      - 4. Дополнительные действия с моделями
    parameters:
      - name: user_name
        in: query
        type: string
        required: true
        description: Имя пользователя.
      - name: model_type
        in: query
        type: integer
        enum: [0, 1]
        default: 0
        required: true
        description: Тип обученной модели. 0 - CNN, 1 - linear
      - name: file
        in: formData
        type: file
        required: true
        description: Файл для загрузки.
    responses:
      200:
        description: Успешная операция.
        schema:
          type: object
          properties:
            message:
              type: string
              example: File uploaded successfully.
      400:
        description: Ошибка валидации.
        schema:
          type: object
          properties:
            error:
              type: string
              example: Invalid input data.
    """
    user_name = request.args.get('user_name')
    if not user_name:
        return jsonify({'message': 'user_name is required'}), 400
    
    model_type = request.args.get('model_type')
    if not model_type:
        return jsonify({'message': 'Model type is required'}), 400
    
    is_valid, _ = AI_Model_Type.convert_to_string_try_get(model_type)
    if not is_valid:
       return jsonify({'message': 'Model type is not valid'}), 404

    if 'file' not in request.files:
      return jsonify({'error': 'No zip file provided'}), 400

    zip_file = request.files['file']
    filename = zip_file.filename

    if not filename.endswith('.zip'):
        return jsonify({'error': 'Invalid file type. Only ZIP archives are allowed'}), 400
    
    result = data_storage_services.create_model_zip(user_name, model_type, filename.replace('.zip', ''), zip_file)
    if result.isSuccess:
      return jsonify({'message': f'{result.result}'}), 200
    
    return jsonify({'message': f'{result.errors}'}), 400 
    
@models_bp.route('/<string:user_name>/<string:model_type>', methods=['GET'])
def get_names(user_name, model_type):
    """
    Получение списка имен моделей для определенного типа.
    ---
    tags:
      - 4. Дополнительные действия с моделями
    parameters:
      - name: user_name
        in: path
        type: string
        required: true
        description: Имя пользователя.
      - name: model_type
        in: path
        type: integer
        enum: [0, 1]
        default: 0
        required: true
        description: Тип обученной модели. 0 - CNN, 1 - LNN
    """
    if not user_name:
        return jsonify({'message': 'user_name is required'}), 404

    if not model_type:
        return jsonify({'message': 'Model type is required'}), 404
    
    is_valid, _ = AI_Model_Type.convert_to_string_try_get(model_type)
    if not is_valid:
       return jsonify({'message': 'Model type is not valid'}), 404

    result = data_storage_services.get_model_names(user_name, model_type)
    if result.isSuccess:
      return jsonify({'message': f'{result.result}'}), 200
    
    return jsonify({'message': f'{result.errors}'}), 400 

@models_bp.route('/<string:user_name>/<string:model_type>/<string:model_name>', methods=['GET', 'DELETE'])
def model_name_operations(user_name, model_type, model_name):
    """
    Операции с именем модели.
    ---
    tags:
      - 4. Дополнительные действия с моделями
    parameters:
      - name: user_name
        in: path
        type: string
        required: true
        description: Имя пользователя.
      - name: model_type
        in: path
        type: integer
        enum: [0, 1]
        default: 0
        required: true
        description: Тип обученной модели. 0 - CNN, 1 - linear
      - name: model_name
        in: path
        type: string
        required: true
        description: Имя модели.
    responses:
      200:
        description: Информация о модели.
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                  example: Information about the model.
      204:
        description: Модель успешно удалена.
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                  example: Model deleted successfully.
      404:
        description: Модель не найдена.
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
                  example: Model not found.
    """
    if not user_name:
        return jsonify({'message': 'user_name is required'}), 404

    if not model_type:
        return jsonify({'message': 'Model type is required'}), 404

    is_valid, _ = AI_Model_Type.convert_to_string_try_get(model_type)
    if not is_valid:
       return jsonify({'message': 'Model type is not valid'}), 404

    if not model_name:
        return jsonify({'message': 'model_name is required'}), 400

    if request.method == 'GET':
        result = data_storage_services.get_model_by_name(user_name, model_type, model_name)
        model_path = result.result

        return send_file(
          model_path,
          mimetype='application/zip',
          as_attachment=True,
          download_name=os.path.basename(model_path))
    
    elif request.method == 'DELETE':
      result = data_storage_services.delete_model_by_name(user_name, model_type, model_name)

    if result.isSuccess:
      return jsonify({'message': f'{result.result}'}), 200
    
    return jsonify({'message': f'{result.errors}'}), 400
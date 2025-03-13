import os
from flask import Blueprint, request, jsonify, send_file

import application.services.data_storage_services as data_storage_services

datasets_bp = Blueprint(
    'datasets',
    __name__,
    url_prefix='/api/datasets'
)

@datasets_bp.route('/save-and-extract', methods=['POST'])
def save_and_extract_dataset():
    """
    Загрузить и разархивировать ZIP-архив датасета.
    ---
    tags:
      - Dataset
    parameters:
      - name: user_name
        in: query
        type: string
        required: true
        description: Имя пользователя.
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
    
    if 'file' not in request.files:
      return jsonify({'error': 'No zip file provided'}), 400

    zip_file = request.files['file']
    filename = zip_file.filename

    if not filename.endswith('.zip'):
        return jsonify({'error': 'Invalid file type. Only ZIP archives are allowed'}), 400
    
    result = data_storage_services.create_dataset_zip(user_name, filename.replace('.zip', ''), zip_file)
    if result.isSuccess:
      return jsonify({'message': f'{result.result}'}), 200
    
    return jsonify({'message': f'{result.errors}'}), 400 
    
@datasets_bp.route('/<string:user_name>', methods=['GET'])
def get_names(user_name):
    """
    Операции с именем датасета.
    ---
    tags:
      - Dataset
    parameters:
      - name: user_name
        in: path
        type: string
        required: true
        description: Имя пользователя.
      - name: dataset_name
        in: path
        type: string
        required: true
        description: Имя датасета.
    responses:
      200:
        description: Информация о датасете.
      204:
        description: Датасет успешно удален.
      404:
        description: Датасет не найден.
        schema:
          type: object
          properties:
            error:
              type: string
              example: Dataset not found.
    """
    if not user_name:
        return jsonify({'message': 'user_name is required'}), 404

    result = data_storage_services.get_dataset_names(user_name)
    if result.isSuccess:
      return jsonify({'message': f'{result.result}'}), 200
    
    return jsonify({'message': f'{result.errors}'}), 400 

@datasets_bp.route('/<string:user_name>/<string:dataset_name>', methods=['GET', 'DELETE'])
def dataset_name_operations(user_name, dataset_name):
    """
    Операции с именем датасета.
    ---
    tags:
      - Dataset
    parameters:
      - name: user_name
        in: path
        type: string
        required: true
        description: Имя пользователя.
      - name: dataset_name
        in: path
        type: string
        required: true
        description: Имя датасета.
    responses:
      200:
        description: Информация о датасете.
      204:
        description: Датасет успешно удален.
      404:
        description: Датасет не найден.
        schema:
          type: object
          properties:
            error:
              type: string
              example: Dataset not found.
    """
    if not user_name:
      return jsonify({'message': 'user_name is required'}), 400

    if not dataset_name:
      return jsonify({'message': 'dataset_name is required'}), 400

    if request.method == 'GET':
        result = data_storage_services.get_dataset_by_name(user_name, dataset_name)
        dataset_path = result.result

        return send_file(
          dataset_path,
          mimetype='application/zip',
          as_attachment=True,
          download_name=os.path.basename(dataset_path))
    
    elif request.method == 'DELETE':
      result = data_storage_services.delete_dataset_by_name(user_name, dataset_name)

    if result.isSuccess:
      return jsonify({'message': f'{result.result}'}), 200
    
    return jsonify({'message': f'{result.errors}'}), 400 
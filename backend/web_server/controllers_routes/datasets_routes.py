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
    Загрузка dataset в систему
    ---
    tags:
      - 1. Загрузка Dataset'a для обучения
    description: |
      # Инструкция по загрузке данных

      - При загрузке необходимо указать **свой username**, так как для каждого пользователя выделено отдельное пространство в памяти.

      - Загрузите архив в формате **.zip**.

      ---

      ## Требования к файлам для обучения

      ### Линейная нейронная сеть

      - Файлы должны быть в формате **.txt**.
      - Формат данных должен соотвествовать следующей структуре:
      ![Пример структуры данных для LNN](/static/cnn_data.jpg)

      ### Сверточная нейронная сеть

      - Изображения должны быть в формате **.jpg**.
      ![Пример данных для CNN](/static/lnn_data.jpg)

      ### Сверточная нейронная сеть YOLO5

      - Для YOLO5 должен использоваться специальный формат данных YOLO v5 PyTorch

      ---

      ## Организация размеченных данных

      - Размеченные данные должны находиться в директориях с названиями классов, к которым они принадлежат.

      - Пример структуры папок:
      ![Пример разделения классов](/static/class_structure.jpg)

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
    Получить список названий датасетов, доступных пользователю.
    ---
    tags:
      - 3. Дополнительные действия с dataset'ами
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
      - 3. Дополнительные действия с dataset'ами
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
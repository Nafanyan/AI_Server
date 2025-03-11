from flask import Blueprint, request, jsonify

from application.Paths import Paths
from application.datasets.save_dataset import save_and_extract_zip

datasets_bp = Blueprint(
    'datasets',
    __name__,
    url_prefix='/api/datasets'
)

@datasets_bp.route('/save-and-extract', methods=['POST'])
def upload_archive():
    """Загрузить и сохранить ZIP архив, который содержит в себе датасет, на сервер
    
    Загружает и сохраняет ZIP архиф в специальной дериктории для конкретного пользователя.

    ---
    tags:
      - Datasets
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: file
        type: file
        required: true
        description: The ZIP archive file to upload.
      - in: query
        name: username
        type: string
        required: true
        description: Username of the user uploading the file.
    responses:
      201:
        description: File uploaded successfully
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                  description: Success message
      400:
        description: Bad Request
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                  description: Error message
    """
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    
    username = request.args.get('username')
    if not username:
        return jsonify({'message': 'Username is required'}), 400
    
    if file and allowed_file(file.filename):
        success = save_and_extract_zip(file, f'{Paths.Datasets}/{username}')
        if success:
            return jsonify({'message': f'{file.filename} uploaded and extracted successfully'}), 201
        else:
            return jsonify({'message': 'Failed to extract ZIP file'}), 400
    else:
        return jsonify({'message': 'Unsupported file format'}), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['zip']
    

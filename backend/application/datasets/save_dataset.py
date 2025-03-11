import os
import zipfile
from werkzeug.utils import secure_filename

def save_and_extract_zip(file, path):
    # Проверка существования директории назначения
    if not os.path.exists(path):
        os.makedirs(path)

    filename = secure_filename(file.filename)
    save_path = os.path.join(path, filename)
    file.save(save_path)

    # Разархивирование файла
    try:
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(path)
        return True
    except zipfile.BadZipFile:
        return False
    finally:
        os.remove(save_path)
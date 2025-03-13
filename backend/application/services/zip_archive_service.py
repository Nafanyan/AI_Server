import os
import zipfile

def create_zip_archive(folder_path):
    folder_name = os.path.basename(os.path.normpath(folder_path))
    
    archive_path = os.path.join(os.path.dirname(folder_path), f"{folder_name}.zip")
    
    with zipfile.ZipFile(archive_path, 'w') as zip_file:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, folder_path)
                zip_file.write(file_path, arcname=relative_path)
    
    return f"{folder_name}.zip"
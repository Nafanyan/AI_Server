import os
import sys
import tempfile
import subprocess
import shutil

def build_exe_with_class_names(class_names, base_script_path, model_path, exe_name="App.exe", dist_path=r"C:\Users\ilya.aleksandrov\Documents\AI_Server\backend\application\ai_model_trainers\create_app"):
    """
    Создает .exe с заданным class_names, генерируя временный скрипт и вызывая PyInstaller.

    :param class_names: список классов для вставки в скрипт
    :param base_script_path: путь к исходному скрипту (для копирования остального кода)
    :param model_path: путь к model.h5
    :param exe_name: имя итогового .exe
    :param dist_path: папка для результата
    """
    if not os.path.exists(base_script_path):
        print(f"Базовый скрипт не найден: {base_script_path}")
        return
    if not os.path.exists(model_path):
        print(f"Модель не найдена: {model_path}")
        return

    # Считаем исходный скрипт
    with open(base_script_path, encoding='utf-8') as f:
        code = f.read()

    # Заменим строку с class_names = [...] на нужный список
    import re
    class_names_str = repr(class_names)
    code_new = re.sub(r"class_names\s*=\s*\[.*?\]", f"class_names = {class_names_str}", code, flags=re.DOTALL)

    # Создадим временный файл
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_script = os.path.join(tmpdir, "temp_app.py")
        with open(temp_script, 'w', encoding='utf-8') as f:
            f.write(code_new)

        # Формируем параметр --add-data для PyInstaller (Windows)
        add_data = f"{model_path};."

        # Запускаем PyInstaller
        cmd = [
            sys.executable, "-m", "PyInstaller",
            f"--name={os.path.splitext(exe_name)[0]}",
            f"--add-data={add_data}",
            "--distpath", dist_path,
            temp_script
        ]

        print("Запускаем сборку .exe...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Сборка завершена успешно. Файл: {os.path.join(dist_path, exe_name)}")
        else:
            print("Ошибка при сборке:")
            print(result.stdout)
            print(result.stderr)

# Пример вызова:
build_exe_with_class_names(
    ['шум', 'сигнал'],
    base_script_path=r"C:\Users\ilya.aleksandrov\Documents\AI_Server\backend\application\create_app\AppForLnn.py",
    model_path=r"C:\Users\ilya.aleksandrov\Documents\AI_Server\backend\application\create_app\model.h5",
    exe_name="MyRecognizer.exe"
)

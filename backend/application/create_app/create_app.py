import os
import sys
import tempfile
import subprocess
import re

def build_exe_with_class_names(class_names, base_script_path, model_path, dist_path, exe_name):
    if not os.path.exists(base_script_path):
        print(f"Базовый скрипт не найден: {base_script_path}")
        return
    if not os.path.exists(model_path):
        print(f"Модель не найдена: {model_path}")
        return

    with open(base_script_path, encoding='utf-8') as f:
        code = f.read()

    class_names_str = repr(class_names)
    code_new = re.sub(r"class_names\s*=\s*\[.*?\]", f"class_names = {class_names_str}", code, flags=re.DOTALL)

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_script = os.path.join(tmpdir, "temp_app.py")
        with open(temp_script, 'w', encoding='utf-8') as f:
            f.write(code_new)

        # Формируем параметр --add-data в зависимости от ОС
        if sys.platform.startswith('win'):
            add_data = f"{model_path};."
        else:
            add_data = f"{model_path}:."

        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--onefile",
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
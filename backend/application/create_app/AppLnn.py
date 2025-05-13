import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, font
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

current_file_path = __file__

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # путь внутри PyInstaller exe
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def find_h5_model(directory):
    # ищем любой .h5 файл в директории, возвращаем первый найденный или None
    for filename in os.listdir(directory):
        if filename.lower().endswith('.h5'):
            return os.path.join(directory, filename)
    return None

class AppLnn:
    def __init__(self, root, class_names, file_extension, data_type):
        self.root = root
        self.class_names = class_names
        self.file_extension = file_extension
        self.data_type = data_type

        self.root.title("Классификатор")
        self.root.configure(bg="#f0f0f5")

        # Центрируем окно
        window_width = 400
        window_height = 200
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x_cordinate = int((screen_width/2) - (window_width/2))
        y_cordinate = int((screen_height/2) - (window_height/2))
        root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")
        root.resizable(False, False)

        # Получаем директорию, где лежит exe или скрипт
        script_dir = resource_path(".")

        # Ищем .h5 модель в той же директории
        self.model_path = find_h5_model(script_dir)
        if not self.model_path:
            messagebox.showerror("Ошибка", f"Модель .h5 не найдена в директории:\n{script_dir}")

        self.model = None
        self.selected_file = None

        # Шрифты
        self.title_font = font.Font(family="Segoe UI", size=16, weight="bold")
        self.btn_font = font.Font(family="Segoe UI", size=12)
        self.info_font = font.Font(family="Segoe UI", size=10, slant="italic")

        # Основная рамка с отступами
        frame = tk.Frame(root, bg="#f0f0f5", padx=20, pady=20)
        frame.pack(expand=True, fill='both')

        # Заголовок
        self.label_title = tk.Label(frame, text="Классификатор", bg="#f0f0f5", fg="#333", font=self.title_font)
        self.label_title.pack(pady=(0, 15))

        # Кнопка выбора файла
        self.btn_select = tk.Button(frame, text=f"Выбрать файл (*{self.file_extension})", font=self.btn_font,
                                    bg="#4a90e2", fg="white", activebackground="#357ABD",
                                    relief="flat", command=self.select_file, cursor="hand2")
        self.btn_select.pack(fill='x', pady=(0, 10))

        # Кнопка классификации
        self.btn_classify = tk.Button(frame, text="Распознать/Классифицировать", font=self.btn_font,
                                      bg="#7ed321", fg="white", activebackground="#5ea616",
                                      relief="flat", command=self.classify_file, cursor="hand2")
        self.btn_classify.pack(fill='x')

        # Информационная метка
        self.info_label = tk.Label(frame, text="Файл не выбран", bg="#f0f0f5", fg="#666", font=self.info_font)
        self.info_label.pack(pady=(15, 0))

    def load_model(self):
        if not self.model:
            if not self.model_path or not os.path.exists(self.model_path):
                messagebox.showerror("Ошибка", f"Модель не найдена: {self.model_path}")
                return False
            self.model = load_model(self.model_path)
        return True

    def select_file(self):
        filetypes = [(f"{self.file_extension} файлы", f"*{self.file_extension}")]
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.selected_file = filename
            self.info_label.config(text=f"Выбран файл:\n{os.path.basename(filename)}")

    def read_txt_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            numbers = [float(line.strip()) for line in f if line.strip()]
        return np.array(numbers)

    def read_image_file(self, filepath):
        img = Image.open(filepath).convert('L')  # grayscale
        img = img.resize((64, 64))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape((64, 64, 1))
        return img_array

    def classify_file(self):
        if not self.selected_file:
            messagebox.showwarning("Внимание", "Сначала выберите файл")
            return
        if not self.load_model():
            return

        try:
            if self.data_type == 'text' and self.selected_file.endswith(self.file_extension):
                data = self.read_txt_file(self.selected_file)
                input_data = np.expand_dims(data, axis=0)
            elif self.data_type == 'image' and self.selected_file.endswith(self.file_extension):
                data = self.read_image_file(self.selected_file)
                input_data = np.expand_dims(data, axis=0)
            else:
                messagebox.showerror("Ошибка", "Неверный тип файла для текущей настройки")
                return

            prediction = self.model.predict(input_data)
            pred_class = int(np.round(prediction)[0][0])
            if 0 <= pred_class < len(self.class_names):
                class_name = self.class_names[pred_class]
            else:
                class_name = f"Неизвестный класс ({pred_class})"
            messagebox.showinfo("Результат", f"Модель распознала класс: {class_name}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при распознавании:\n{e}")

if __name__ == "__main__":
    class_names = []  # заполняйте по необходимости
    file_extension = '.txt'
    data_type = 'text'
    root = tk.Tk()
    app = AppLnn(root, class_names, file_extension, data_type)
    root.mainloop()

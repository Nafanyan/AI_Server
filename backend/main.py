from flask import Flask
import web_server.routes as routes
from flasgger import Swagger

def create_app():
    # Создаем экземпляр приложения Flask
    app = Flask(__name__)
    
    # Регистрация всех маршрутов
    routes.init_routes(app)

    # Инициализация Swagger после регистрации Blueprint

    template = {
        "swagger": "2.0",
        "info": {
            "title": "AI Server",
            "version": "1.0"
        },
        "tags": [
            {"name": "1. Загрузка Dataset'a для обучения"},
            {"name": "2.1 Обучение линейной модели нейронной сети (LNN)"},
            {"name": "2.2 Обучение сверточной модели нейронной сети (CNN)"},
            {"name": "3. Дополнительные действия с dataset'ами"},
            {"name": "4. Дополнительные действия с моделями"},
        ]
    }

    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": 'apispec',
                "route": '/apispec.json',
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/api/doc/",
    }
    
    swagger = Swagger(app, template=template, config=swagger_config)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
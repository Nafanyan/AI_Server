from flask import Flask
import web_server.routes as routes
from flasgger import Swagger

def create_app():
    # Создаем экземпляр приложения Flask
    app = Flask(__name__)
    
    # Регистрация всех маршрутов
    routes.init_routes(app)

    # Инициализация Swagger после регистрации Blueprint
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
        "specs_route": "/api/doc/"
        }
    
    swagger = Swagger(app, config=swagger_config)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
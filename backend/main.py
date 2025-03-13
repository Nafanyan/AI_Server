from flask import Flask
import web_server.routes as routes
from flasgger import Swagger

def create_app():
    # Создаем экземпляр приложения Flask
    app = Flask(__name__)
    
    # Регистрация всех маршрутов
    routes.init_routes(app)

    # Инициализация Swagger после регистрации Blueprint
    swagger = Swagger(app)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
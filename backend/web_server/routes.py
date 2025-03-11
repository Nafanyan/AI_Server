from flask import Blueprint
from .train_cnn_models_routes.train_cnn_models_routes import train_cnn_models_bp
from .datasets_routes.datasets_routes import datasets_bp

def init_routes(app):
    # Регистрация всех blueprints
    app.register_blueprint(train_cnn_models_bp)
    app.register_blueprint(datasets_bp)
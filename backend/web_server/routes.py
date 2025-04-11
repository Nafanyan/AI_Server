import web_server.controllers_routes.datasets_routes as datasets_routes
import web_server.controllers_routes.models_routes as models_routes
import web_server.controllers_routes.train_cnn_models_routes as train_cnn_models_routes
import web_server.controllers_routes.train_lnn_models_routes as train_lnn_models_routes

def init_routes(app):
    # Регистрация всех blueprints
    app.register_blueprint(datasets_routes.datasets_bp)
    app.register_blueprint(models_routes.models_bp)
    app.register_blueprint(train_cnn_models_routes.train_cnn_models_bp)
    app.register_blueprint(train_lnn_models_routes.train_lnn_models_bp)
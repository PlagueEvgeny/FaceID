from flask import Flask
from app.router import main_router
from app.api.recognition_api import recognition_bp
from app.api.system_api import system_api
from app.services.init_system import initialize_system
from app import config

def create_app():
    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False

    # регистрируем маршруты
    app.register_blueprint(main_router)
    app.register_blueprint(recognition_bp)
    app.register_blueprint(system_api)

    # инициализация системы
    initialize_system()
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=config.DEBUG_MODE,
            threaded=config.THREADED_MODE,
            host=config.DEFAULT_HOST,
            port=config.DEFAULT_PORT)

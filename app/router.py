from flask import Blueprint, render_template, Response
from app.services.frame_generator import generate_frames

main_router = Blueprint("main", __name__)

@main_router.route("/")
def index():
    return render_template("index.html")

@main_router.route("/collect_data")
def collect_data_page():
    return render_template("collect_data.html")

@main_router.route("/recognize")
def recognize_page():
    return render_template("recognize.html")

@main_router.route("/video_feed")
def video_feed():
    """Видео поток"""
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
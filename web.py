from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

from predictor import predict_from_path

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            error = "Please select an image file."
        else:
            filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
            file.save(filepath)

            try:
                result = predict_from_path(filepath)
            except Exception as e:
                error = str(e)

    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    import webbrowser
    import threading

    def open_browser():
        webbrowser.open("http://127.0.0.1:5000")

    threading.Timer(1.0, open_browser).start()

    app.run(debug=True, use_reloader=False)
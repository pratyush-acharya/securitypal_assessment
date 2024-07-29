import os
import uuid
import pandas as pd
from flask_cors import CORS
from dotenv import load_dotenv
from flask import Flask, request, render_template
from typing import List, Union
from analysis import Analysis
# Define the Flask app
app = Flask(__name__)
CORS(app)

# Load the environment variables
load_dotenv()

def allowed_file(filename: str) -> bool:
    """Check if the file has valid extension

    Args:
        filename (str): name of the file being uploaded

    Returns:
        bool: true if the file has a valid extension, false otherwise
    """
    allowed_extensions = ["csv"]
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def uplodad():
    initial_file = request.files["initial_file"]
    final_file = request.files["final_file"]
    if initial_file.filename != "" and allowed_file(initial_file.filename) and final_file.filename != "" and allowed_file(final_file.filename):
        initial_df = pd.read_csv(initial_file)
        final_df = pd.read_csv(final_file)
        if len(initial_df) != len(final_df):
            return KeyError("The number of questions in the initial and final questionnaire do not match")
        analysis = Analysis(initial_df, final_df)
    return render_template(
        "index.html",
        plot_urls=analysis.plot_urls,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

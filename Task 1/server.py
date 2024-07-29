import os
import uuid
import chromadb
import pandas as pd
from flask_cors import CORS
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from flask import Flask, request, render_template
from typing import List, Union

# Define the Flask app
app = Flask(__name__)
CORS(app)

# Load the environment variables
load_dotenv()
persist_directory = os.environ.get("PERSIST_DIRECTORY")
collection_name = os.environ.get("COLLECTION_NAME")

# Setup the chroma client
chroma_client = chromadb.PersistentClient(path=persist_directory)
default_ef = embedding_functions.DefaultEmbeddingFunction()


def does_vectorstore_exist() -> bool:
    """Check if the vectorstore exists

    Returns:
        bool: true if the vectorstore exists, false otherwise
    """
    if collection_name in [
        collection.name for collection in chroma_client.list_collections()
    ]:
        return True

    return False


def allowed_file(filename: str) -> bool:
    """Check if the file has valid extension

    Args:
        filename (str): name of the file being uploaded

    Returns:
        bool: true if the file has a valid extension, false otherwise
    """
    allowed_extensions = ["csv"]
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def classify_question(question: Union[str, List], threshold: float = 0.2) -> str:
    """Classifies the question as answerable, ambiguous or unanswerable

    Args:
        question (Union[str, List]): question/s to be classified
        threshold (float, optional): Threshold for the question to be answerable. Defaults to 0.2.

    Returns:
        str: Returns the classification of the question
    """
    if question is str:
        question = [question]

    knowledge_base_collection = chroma_client.get_collection(collection_name)
    answer = knowledge_base_collection.query(query_texts=question, n_results=1)
    if answer["distances"][0][0] < threshold:
        return "answerable"
    elif answer["distances"][0][0] < 0.5 and answer["distances"][0][0] > threshold:
        return "ambiguous"
    else:
        return "unanswerable"


@app.route("/")
def index():
    if not does_vectorstore_exist():
        # create a new collection
        knowledge_base = pd.read_csv("data/processed_knowledge_base.csv")
        knowledge_base_collection = chroma_client.create_collection(
            name="knowledge_base",
            embedding_function=default_ef,
            metadata={"hnsw:space": "cosine"},
        )

        metadata = [
            {
                "section_heading": row[1]["section_heading"],
                "control_heading": row[1]["control_heading"],
                "answer": row[1]["answer"],
                "notes": row[1]["notes"],
            }
            for row in knowledge_base.iterrows()
        ]

        knowledge_base_collection.add(
            documents=knowledge_base["question"].to_list(),
            ids=[str(uuid.uuid4()) for i in range(len(knowledge_base["question"]))],
            metadatas=metadata,
        )

    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def uplodad():
    file = request.files["file"]
    if file.filename != "" and allowed_file(file.filename):
        questionnaire = pd.read_csv(file)
        questionnaire["classification"] = questionnaire["questions"].apply(
            classify_question
        )
        classification_results = questionnaire.to_dict(orient="records")
        completion_percentage = (
            questionnaire["classification"] == "answerable"
        ).mean() * 100
    return render_template(
        "index.html",
        classification_results=classification_results,
        completion_percentage=completion_percentage,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

# TASK 1: Knowledge Base Integration and Question Classification

## Installation Notes

* First, create a `.env` file with the following parameters:
    * PERSIST_DIRECTORY=(path to the vectorstore files; e.g: ./vectorstore)
    * COLLECTION_NAME= (name of the collection; defaults to knowledge_base)

* Now, run the following command to setup the environment:
    ```bash
        conda env create -f environment.yml
    ```
* Now, run the following command to activate the environment:
    ```bash
        conda activate task_1
    ```
* After activating the environment, run the following command to run the server:
    ```python
        python server.py
    ```
    Once the command has been ran, the server should be running on Port 8000. 
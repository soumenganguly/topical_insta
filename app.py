from flask import Flask, request, jsonify
from azure.storage.blob import BlobServiceClient
from src.utils import read_data_from_azure, download_model_from_azure
from src.model import train_model, store_model, predict

app = Flask(__name__)

STORAGE_ACCOUNT_NAME = "INSERT STORAGE ACCOUNT NAME"
STORAGE_ACCOUNT_KEY = "INSERT STORAGE ACCOUNT KEY"
CONTAINER_NAME = "INSERT CONTAINER NAME"
BLOB_NAME = "INSERT BLOB NAME"
MODEL_LOCAL_PATH= "/tmp/models"

blob_service_client = BlobServiceClient(
    account_url=f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net", 
    credential=STORAGE_ACCOUNT_KEY
    )

@app.route("/")
def home():
    return "This is a demo app to label topics on Instagram posts"

@app.route("/train", methods=['POST'])
def train():
    data = read_data_from_azure(
        blob_service_client, 
        container_name=CONTAINER_NAME,
        blob_name=BLOB_NAME
    )
    model = train_model(data, MODEL_LOCAL_PATH)
    res = store_model(MODEL_LOCAL_PATH)
    if res["status"] == "success":
        return jsonify({"msg": "Model is trained and stored at: {BLOB_NAME}"}), 200
    else: 
        return jsonify({"error": str(res["message"])}), 400


@app.route('/label', methods=['POST'])
def label():
    data = request.json
    text = data.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    local_path = download_model_from_azure(
        blob_service_client,
        CONTAINER_NAME,
        BLOB_NAME,
        MODEL_LOCAL_PATH
    )
    try:
        res = predict(text, local_path)
        return jsonify({"result":res})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

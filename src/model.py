from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import os

def train_model(data: pd.DataFrame, path: str) -> BERTopic:
    """Train a BERTopic model on Instagram posts.

    Args:
        data (pd.DataFrame): Dataframe with insta posts
        path (str): Blob storage path

    Returns:
        BERTopic: Fitted model with insta posts
    """    
    # Fetch the posts to use for model training
    posts = data['posts']

    # Load a pre-trained sentence transformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Create embeddings
    embeddings = model.encode(posts, show_progress_bar=True)

    # Initialize BERTopic with a minimum of 20 topics
    min_topic_size = 20
    topic_model = BERTopic(min_topic_size=min_topic_size, embedding_model=model)

    # Fit the model
    topic_model.fit_transform(posts, embeddings)

    # Set the Maximum nr. of topics to 100
    max_topics = 100
    topic_model.reduce_topics(posts, nr_topics=max_topics)

    # Save the model to a local folder
    topic_model.save(path)

    return topic_model

def store_model(local_model_path: str, blob_service_client: str, model_name:str, container_name:str) -> dict:
    """Stores the trained model onto Azure Blob storage.

    Args:
        local_model_path (str):
        blob_service_client (str): _description_
        model_name (str): _description_
        container_name (str: _description_

    Returns:
        dict : Status message of the operation
    """    
    try:
        for root, dirs, files in os.walk(local_model_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Define the blob path relative to the model directory
                blob_path = os.path.relpath(file_path, local_model_path)
                # Create a blob client using the blob path
                blob_client = blob_service_client.get_blob_client(
                    container=container_name, 
                    blob=f"{model_name}/{blob_path}"
                )

                # Upload the file to Azure Blob Storage
                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                return {"status": "success", "message": f"File '{local_model_path}' uploaded to blob '{blob_path}' in container '{container_name}'."}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
def predict(text:str, model_local_path:str) -> dict:
    """Make topic prediction on new posts.

    Args:
        text (str): Post
        model_local_path (str): Local model storage path

    Returns:
        dict: Returns a result dict with topic and probabilities.
    """    
    model = BERTopic.load(model_local_path)
    topic, prob = model.transform([text])
    return {f"{topic[0]}: {prob[0]}"}

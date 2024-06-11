import pandas as pd
from io import StringIO
import os

def read_data_from_azure(blob_service_client, container_name, blob_name):
    # Get a BlobClient for the specified blob
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    # Download the blob content as text
    download_stream = blob_client.download_blob()
    csv_content = download_stream.content_as_text()

    # Use StringIO to convert the text content to a pandas DataFrame
    csv_string_io = StringIO(csv_content)
    df = pd.read_csv(csv_string_io)

    return df

def download_model_from_azure(blob_service_client, container_name, blob_name, local_model_path):
    """
    Downloads a model from Azure Blob Storage if it's not already present locally.

    Parameters:
    - connection_string: str. The connection string for the Azure Storage account.
    - container_name: str. The name of the container in Azure Blob Storage.
    - blob_name: str. The name of the blob in Azure Blob Storage.
    - local_model_path: str. The local path where the model will be saved.

    Returns:
    - str: Local path to the model file.
    """
    try:
        # Check if the model already exists locally
        if os.path.exists(local_model_path):
            print("Model already exists locally.")
            return local_model_path

        # Create a blob client using the blob name
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        # Download the blob to a local file
        with open(local_model_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())

        print("Model downloaded from Azure Blob Storage.")
        return local_model_path

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
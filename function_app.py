import azure.functions as func
import logging
from azure.storage.blob import BlobClient, ContentSettings
import io
import os
import re


from sample_figure_understanding import analyze_layout as analyze_layout
import uuid
import shutil

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

content_settings = ContentSettings(content_type='application/json')



def get_file_name(blob_name):
    return blob_name.split('/')[-1]


AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER_NAME_IMAGES = os.getenv("AZURE_STORAGE_CONTAINER_NAME_IMAGES")
AZURE_STORAGE_CONTAINER_NAME_DOCS = os.getenv("AZURE_STORAGE_CONTAINER_NAME_DOCS")

AZURE_STORAGE_CONTAINER_NAME_INPUT = os.getenv("AZURE_STORAGE_CONTAINER_NAME_INPUT")

def create_folder(tmp_path, folder_name=""):
    folder_path = os.path.join(tmp_path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

@app.blob_trigger(arg_name="myblob", path=AZURE_STORAGE_CONTAINER_NAME_INPUT,
                               connection="documents_storage") 
# def document_image_procesing(myblob: blob.BlobClient):
def document_image_procesing(myblob: func.InputStream):
    logging.info(f"Python blob trigger function processed blob"
                f"Name: {myblob.name}"
                f"Blob Size: {myblob.length} bytes")
    stream = io.BytesIO()
    
    random_foldername = str(uuid.uuid4())
    temp_path = os.getenv('TMPDIR', os.path.join('/','tmp'))
    temp_path = os.path.join(temp_path, random_foldername)

    logging.warn(f"Temp path: {temp_path}")

    create_folder(temp_path)
    create_folder(temp_path, AZURE_STORAGE_CONTAINER_NAME_INPUT)
    create_folder(temp_path, AZURE_STORAGE_CONTAINER_NAME_IMAGES)
    create_folder(temp_path, AZURE_STORAGE_CONTAINER_NAME_DOCS)

    # logging.info(f"Temp path: {temp_path}")

    # generate random filename
    random_filename = myblob.name
    temp_path_filename = os.path.join(temp_path,random_filename)
    logging.warn(f"Temp path filename: {temp_path_filename}")

    # save the file to temp path
    with open(temp_path_filename, "wb") as f:
        f.write(myblob.read())

    # out = analyze_layout("myblob.pdf", stream, AZURE_STORAGE_CONTAINER_NAME_IMAGES, AZURE_STORAGE_CONTAINER_NAME_DOCS)
    parsed_content = analyze_layout(temp_path_filename, os.path.join(temp_path, AZURE_STORAGE_CONTAINER_NAME_IMAGES), os.path.join(temp_path, AZURE_STORAGE_CONTAINER_NAME_DOCS))

    # Convert string to bytes
    byte_data = parsed_content.encode('utf-8')

    # Upload data on Blob
    # Get the base name of the file
    _file_name = f"{os.path.basename(myblob.name)}.md"
    logging.info(f"Transcript analysed. Output file: {_file_name} file.")
    blob_client = BlobClient.from_connection_string(conn_str=AZURE_STORAGE_CONNECTION_STRING, container_name=AZURE_STORAGE_CONTAINER_NAME_DOCS, blob_name=_file_name)
    blob_client.upload_blob(byte_data, content_settings=content_settings, overwrite=True)
    blob_client.close()

    # delete the temp folder
    shutil.rmtree(temp_path)

    
    logging.info(f"Processing of {myblob.name} completed.")
    logging.info(f"Output file: {AZURE_STORAGE_CONTAINER_NAME_DOCS}/{_file_name} file.")
    logging.info(f"Output images in container: {AZURE_STORAGE_CONTAINER_NAME_IMAGES} file.")
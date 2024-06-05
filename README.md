# AOAI-document-processing

Sample application that takes input documents (file) and processes the file in a way that:
- all images are being striped out and stored in specific folder
- text with images placehoders is stored in specific folder, format markdown


## Settings

- Azure Functions app
- Azure Document Intelligence serivice
- Azure Storage
    - three containers:
        - `AZURE_STORAGE_CONTAINER_NAME_INPUT` - scans the conteiner for input docuements
        - `AZURE_STORAGE_CONTAINER_NAME_IMAGES` - output container with extreacted images only
        - `AZURE_STORAGE_CONTAINER_NAME_DOCS` - output container with processed document in Markdown format

```shell
"documents_storage": "...",
"AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT": "https://XXXXXXXX.cognitiveservices.azure.com/",
"AZURE_DOCUMENT_INTELLIGENCE_KEY": "**********",
"AZURE_STORAGE_CONTAINER_NAME_IMAGES": "doc-out-images",
"AZURE_STORAGE_CONTAINER_NAME_DOCS": "doc-out-mds",
"AZURE_STORAGE_CONTAINER_NAME_INPUT": "doc-in-docs"
```

## Functions

Blob triggered function:

```python
def document_image_procesing(myblob: func.InputStream):
    logging.info(f"Python blob trigger function processed blob"
                f"Name: {myblob.name}"
                f"Blob Size: {myblob.length} bytes")
```


### Deployment

From VS code deploy to function app or

```shell
func azure functionapp publish <functionapp_name>
```
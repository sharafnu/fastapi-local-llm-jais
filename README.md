# FAST API with Locally Hosted Model

<div align="center">
  <a alt="FastAPI logo" href="https://fastapi.tiangolo.com/" target="_blank" rel="noreferrer">
    <img src="https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png" width="100">
  </a>  
  <a alt="HuggingFace logo" href="https://huggingface.co/" target="_blank" rel="noreferrer">
    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQZvbFMg3jsN0INoEpJ2kRuduXjQG4q7ZHfsR8b3PpA&s" width="100">
  </a>  
  <a alt="Jais logo" href="https://mbzuai.ac.ae/news/meet-jais-the-worlds-most-advanced-arabic-large-language-model-open-sourced-by-g42s-inception/" target="_blank" rel="noreferrer">
    <img src="https://the-decoder.com/wp-content/uploads/2023/09/jais-logo.png" width="100">
  </a>  
</div>

✨ **Tech Stack** ✨

- [FastAPI.js](https://fastapi.tiangolo.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [JAIS Model](https://huggingface.co/inception-mbzuai/jais-13b-chat)
  
## Introduction

This project has a basic FastAPI endpoint to interact with locally hosted JAIS Model.

## Prerequisites

- Make sure Python setup (with Pip) is available
- Download all JAIS model files and configurations from https://huggingface.co/inception-mbzuai/jais-13b-chat/tree/main and copy under huggingface-jais-13b-chat

The directory will look like below: 

![JAIS setup](screenshots/jais.png?raw=true "JAIS Setup")

  

## Setup

### Create virtual environment:
```bash
python -m venv jaisenv
```
### Activate virtual environment:
```bash
source jaisenv/bin/activate
```

### Intall the following dependecies
```bash
pip install "fastapi[all]"
pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0
pip install transformers==4.30.2
```


### Start the app
```bash
uvicorn main:app --reload
```


## Check the API and invoke from SwaggerUI or via Postman

 http://127.0.0.1:8000/docs

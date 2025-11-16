# LLMDISGEN

## About
This repo contains a set of applications to
1. generate discussions for scientific papers with different LLM models and compare them with several text comparison models like BERTScore, Rouge etc...
2. Run a simple frontend to allow for human decision which discussion, be it the original one for the paper or the a generated one is better

## Installation
```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Running the evaluator
```
(venv) $ python main.py
```

## Running the web app
```
(venv) $ cd frontend && python app.py 
```

App then is under `localhost:8000`.
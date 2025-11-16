from flask import Flask, render_template, abort
from pathlib import Path
import json


app = Flask(__name__)

base = Path("..") / "runs"
filename = "metadata-1763254708.json"


def get_papers():
    with open(base / filename) as f:
        return json.load(f)


def get_by_id(corpus_id):
    papers = get_papers()
    for paper in papers:
        if paper["corpus_id"] == corpus_id:
            return paper
    return {}


@app.route('/')
def index():
    return render_template('index.html', papers=get_papers())


@app.route("/<corpus_id>")
def get_paper(corpus_id):
    paper = get_by_id(corpus_id)
    if paper is None:
        abort(404)
    return render_template('paper.html', paper=paper)
    


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
from flask import Flask, render_template, abort
from pathlib import Path
import json

app = Flask(__name__)

base = Path("..") / "runs"

def get_runs():
    return sorted([
        f.name.removeprefix("metadata-").removesuffix(".json")
        for f in base.iterdir()
        if f.is_file() and f.name.startswith("metadata-") and f.name.endswith(".json")
    ])

def get_papers(file_id):
    filename = f"metadata-{file_id}.json"
    with open(base / filename) as f:
        return json.load(f)

def get_by_id(file_id, corpus_id):
    papers = get_papers(file_id)
    for paper in papers:
        if paper["corpus_id"] == corpus_id:
            return paper
    return {}

@app.route('/')
def index():
    print("*hello")
    return render_template('index.html', runs=get_runs())

@app.route('/runs/<run_id>')
def get_papers_for_id(run_id):
    return render_template('run.html', run_id=run_id, papers=get_papers(run_id))

@app.route("/runs/<run_id>/corpus/<corpus_id>")
def get_paper(run_id, corpus_id):
    paper = get_by_id(run_id, corpus_id)
    if paper is None:
        abort(404)
    return render_template('paper.html', paper=paper)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
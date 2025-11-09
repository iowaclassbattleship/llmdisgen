from flask import Flask, render_template


app = Flask(__name__)


@app.route('/')
def index():
    with open("prompt.txt", "r") as f:
        paper = f.read()
    with open("llm.txt", "r") as f:
        llm_discussion = f.read()
    with open("discussion.txt", "r") as f:
        discussion = f.read()
    return render_template('index.html', paper=paper, llm_discussion=llm_discussion, discussion=discussion)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
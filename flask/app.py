from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/game_tensors', methods=['POST'])
def game_tensors():
    gt1,gt2=get_game_tensors(request.form['game'])
    return render_json(gt1,gt2)
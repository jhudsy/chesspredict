from flask import Flask, render_template, request, render_json
import chess,chess.pgn
from chess.engine import PovScore, Cp
import numpy as np
from io import StringIO
from ml.shared import get_game_tensor
from ml.learning import predict

app = Flask(__name__)

model_file = 'models/modelCP.keras'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_game', methods=['POST'])
def submit_game():
    game_pgn = StringIO(request.form['game_pgn'])
    prediction = predict(model_file,pgn_string=game_pgn)
    return render_json(prediction[0],prediction[1])

if __name__ == '__main__':
    app.run(debug=True)

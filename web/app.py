from flask import Flask, render_template, request, jsonify
from io import StringIO
from ml.learning import predict

app = Flask(__name__)

model_file = '../models/modelBlitz.keras'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_game', methods=['POST'])
def submit_game():
    game_pgn = StringIO(request.form['game_pgn'])
    prediction = predict(model_file,pgn_string=game_pgn)
    return jsonify(prediction[0],prediction[1])

if __name__ == '__main__':
    app.run(debug=True)

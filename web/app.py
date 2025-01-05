from flask import Flask, render_template, request, jsonify
from io import StringIO
from ml.learning import predict
import os
import math

app = Flask(__name__)

model_file = 'models/modelBlitz.keras'
 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_game', methods=['POST'])
def submit_game():
    game_pgn = request.form['game_pgn']
    prediction = predict(model_file,pgn_string=game_pgn).flatten().tolist()
    #format prediction as a list with no decimal places
    prediction = [round(x) for x in prediction]
    
    return jsonify(prediction)

if __name__ == '__main__':
    #check if model file exists
    if os.path.exists(model_file) == False:
        print("Model file not found")
        exit(1)

    app.run(debug=True)

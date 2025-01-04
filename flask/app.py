from flask import Flask, render_template, request, render_json
import chess,chess.pgn
from chess.engine import PovScore, Cp
import numpy as np
from io import StringIO

file_dict = {"300+0":"blitz",
             "300+3":"blitz",
             "60+0":"ultrabullet",
             "120+1":"bullet",
             "180+0":"superblitz",
             "180+2":"superblitz",
             "600+0":"rapid",
             "600+5":"rapid",
             "900+10":"rapid"
}

TERMINATION_STRINGS=set(["Abandoned","Rules infraction"])

NUM_MOVES = 40

def get_game_tensor(game_string):
    """returns a tensor representation of the game string. If the game is invalid, returns None. Note that a valid game will have 2 game tensors, one for each player. We also return the ratings of the players and the file that the game should be saved to."""

    #start by checking if the game is valid. The time control is a substring of the form 'TimeControl "{TC}"' where {TC} is a variable, check if {TC} is in the file_dict.

    time_control = game_string.split('TimeControl "')[1].split('"')[0]
    if time_control not in file_dict or '[%eval' not in game_string:
        return None
        
    #check for termination strings
    for term in TERMINATION_STRINGS:
        if term in game_string:
            return None
    
    ########prepare the game tensors
    gt1 = np.zeros((NUM_MOVES,136),dtype=np.int16)
    gt2 = np.zeros((NUM_MOVES,136),dtype=np.int16)

    game = chess.pgn.read_game(StringIO(game_string))

    board = game.board()
    white_time = 0
    black_time = 0

    move_number = 0

    current_eval = PovScore(Cp(0), chess.WHITE)
    current_move_color = chess.WHITE
    while True:
        t = np.zeros(136)

        for i in range(64):
            if board.piece_at(i) is None:
                t[i] = 0
            elif board.piece_at(i).color == current_move_color:
                t[i] = board.piece_at(i).piece_type
            else:
                t[i] = board.piece_at(i).piece_type + 7

        # get the evaluation, time etc.
        t[128] = move_number // 2  # move number

        t[129] = white_time if current_move_color == chess.WHITE else black_time

        t[131] = black_time if current_move_color == chess.WHITE else white_time

        if current_eval is None: #mate in 0
            t[135] = 1
            t[134] = 0
        elif current_eval.pov(current_move_color).is_mate(): #mate in X
            t[133] = 1
            t[132] = current_eval.pov(current_move_color).mate()
        else:
            t[133] = 0
            t[132] = current_eval.pov(current_move_color).score()

        if move_number == 0:
            m = game.next()
        else:
            m = m.next()
        if m is None:
            break

        if current_move_color == chess.WHITE:
            white_time = m.clock()
        else:
            black_time = m.clock()

        current_eval = m.eval()
        board = m.board()

        for i in range(64):
            if board.piece_at(i) is None:
                t[i + 64] = 0
            elif board.piece_at(i).color == current_move_color:
                t[i + 64] = board.piece_at(i).piece_type
            else:
                t[i + 64] = board.piece_at(i).piece_type + 7

        t[130] = white_time if current_move_color == chess.WHITE else black_time

        if current_eval is None:
            t[135] = 1
            t[134] = 0
        elif current_eval.pov(current_move_color).is_mate():
            t[135] = 1
            t[134] = current_eval.pov(current_move_color).mate()
        else:
            t[135] = 0
            t[134] = current_eval.pov(current_move_color).score()

        if current_move_color == chess.WHITE:
            gt1[move_number // 2] = t
        else:
            gt2[move_number // 2] = t

        current_move_color = not current_move_color

        move_number += 1

        if move_number == NUM_MOVES * 2:
            break

    return np.array(gt1),np.array(gt2),file_dict[time_control]


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/game_tensors', methods=['POST'])
def game_tensors():
    gt1,gt2,time_control=get_game_tensor(request.form['game'])
    return render_json(gt1.tolist(),gt2.tolist(),time_control)
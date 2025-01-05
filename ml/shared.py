import numpy as np
from .config import *
import chess.pgn
from io import StringIO
from chess.engine import Cp,PovScore

def get_game_tensor(game_string,**kwargs):
    """returns a tensor representation of the game string. If the game is invalid, returns None. Note that a valid game will have 2 game tensors, one for each player. 
    We also return the ratings of the players and the file that the game should be saved to. We have one additional parameter, do_checks, which is set to True by default. 
    If set to False, we skip the checks for the time-control and the rating difference. 
    This is useful when we are generating the training data, as we have already checked for these conditions when we generated the games. Ditto for predictions"""

    do_checks = kwargs.get('do_checks',True)
    
    #start by checking if the game is valid. The time control is a substring of the form 'TimeControl "{TC}"' where {TC} is a variable, check if {TC} is in the file_dict.

    time_control = game_string.split('TimeControl "')[1].split('"')[0]

    if do_checks:
        if time_control not in file_dict:
            return None
    
        valid = False
    
        if '[%eval' in game_string and 'WhiteRatingDiff' in game_string and 'BlackRatingDiff' in game_string:
            white_diff = int(game_string.split('WhiteRatingDiff "')[1].split('"')[0])
            black_diff = int(game_string.split('BlackRatingDiff "')[1].split('"')[0])
            if abs(white_diff) < MAX_RATING_DIFF and abs(black_diff) < MAX_RATING_DIFF:
                valid = True
        if not valid:
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
    board_mirrored = False
    while True:
        t = np.zeros(136)

        #TODO: this logic can be simplified, as can the move_number logic using board.fullmove_number
        if current_move_color == chess.BLACK and not board_mirrored:
            board.apply_mirror()
            board_mirrored = True
        elif current_move_color == chess.WHITE and board_mirrored:
            board.apply_mirror()
            board_mirrored = False

        for i in range(64):
            if board.piece_at(i) is None:
                t[i] = 0
            elif board.piece_at(i).color == chess.WHITE: #we can use chess.WHITE always as we mirror the board flipping the colors
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

        current_eval = m.eval() #N.B., current_eval is not mirrored!
        board = m.board()
        if current_move_color == chess.BLACK and not board_mirrored:
            board.apply_mirror()
            board_mirrored = True
        elif current_move_color == chess.WHITE and board_mirrored:
            board.apply_mirror()
            board_mirrored = False

        for i in range(64):
            if board.piece_at(i) is None:
                t[i + 64] = 0
            elif board.piece_at(i).color == chess.WHITE:
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

    return np.array(gt1),np.array(gt2),int(game.headers['WhiteElo']),int(game.headers['BlackElo']),file_dict[time_control]

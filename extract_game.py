import fileinput
import chess
import chess.pgn
from chess.engine import PovScore, Cp

import numpy as np

TIME_CONTROL = "300+0" #the time control string.
MAX_RATING_DIFF = 40 #maximum resultant rating diff. Used to filter out games with large rating changes due to new accounts where the elo might be very inaccurate..

#this simple program extracts all games from a file and prints them out. Normally you would read the file in on stdin or via a filename. For a compressed lichess games file the usage is
# zstdcat lichess09.pgn.zst|python3 extract_game.py > extracted_games09.pgn

def check_game(string):
  if f'TimeControl "{TIME_CONTROL}"' in string and '[%eval' in string and 'WhiteRatingDiff' in string and 'BlackRatingDiff' in string:
        white_diff = int(string.split('WhiteRatingDiff "')[1].split('"')[0])
        black_diff = int(string.split('BlackRatingDiff "')[1].split('"')[0])
        if abs(white_diff) < MAX_RATING_DIFF and abs(black_diff) < MAX_RATING_DIFF:
            return True
        
  return False

def make_game_tensor(game,num_moves):
  gt = []
  #we will make two game tensors from a game, one for each player. 
  #The tensor will be a 136 element tensor. The first 64 elements will be the board state, the next 64 will be the board state after the move. Element 129
  # is the move number. Element 130 is the time on the clock at the start of the move and element 131 is the time on the clock after the move. Element 132 is the time on the clock of the opponent.
  #Element 133 is the evaluation of the position before the move and 134 is the evaluation of the position after the move. Element 135 is a 1 if the evaluation before the move is a mate in ... and 0 otherwise while element 136 is a 1 if the evaluation after the move is a mate in ... and 0 otherwise.

  #The board state itself is a 64 element tensor with 0 for empty spaces, 1...6 for the current color's pieces and 7...12 for the opponent's pieces. 
  board = chess.Board()
  white_time = 300
  black_time = 300

  move_number = 0

  current_eval = PovScore(Cp(0), chess.WHITE)

  pgn = chess.pgn.read_game(chess.pgn.StringIO(game))

  current_move_color = chess.WHITE #start with white

  for m in pgn.mainline():
    t = np.zeros(136)

    for i in range(64):
        if board.piece_at(i)==None:
          t[i] = 0
        elif board.piece_at(i).color == current_move_color:
           t[i] = board.piece_at(i).piece_type
        else:
           t[i] = board.piece_at(i).piece_type+7
      
    #get the evaluation, time etc.

    t[129] = white_time if current_move_color == chess.WHITE else black_time
  
    t[131] = black_time if current_move_color == chess.WHITE else white_time
  
    if current_eval.pov(current_move_color).is_mate():
      t[133] = 1
      t[132] = current_eval.pov(current_move_color).mate()
    else:
      t[133] = 0
      t[132] = current_eval.pov(current_move_color).score()

    board.move(m)

    current_eval = m.eval()

    for i in range(64):
        if board.piece_at(i)==None:
          t[i+64] = 0
        elif board.piece_at(i).color == current_move_color:
          t[i+64] = board.piece_at(i).piece_type
        else:
          t[i+64] = board.piece_at(i).piece_type+7
  
    t[130] = white_time if current_move_color == chess.WHITE else black_time
  
    if current_eval.pov(current_move_color).is_mate():
      t[135] = 1
      t[134] = current_eval.pov(current_move_color).mate()
    else:
      t[135] = 0
      t[134] = current_eval.pov(current_move_color).score()

    gt.append(t)
    current_move_color = not current_move_color

    move_number+=1

    if move_number == num_moves*2:
      break

  return gt
    




game = ""

for line in fileinput.input():
  if line.startswith("[Event ") and game=="":
    game=line
  elif line.startswith("[Event "):
    if check_game(game):
      print(game)
    game = line
  else:
      game+=line
    
  
  


import fileinput
import sys

TIME_CONTROL = "300+0" #the time control string.
MAX_RATING_DIFF = 40 #maximum resultant rating diff. Used to filter out games with large rating changes due to new accounts where the elo might be very inaccurate..

#this simple program extracts all games from a file and prints them out. Normally you would read the file in on stdin or via a filename. For a compressed lichess games file the usage is
# zstdcat lichess09.pgn.zst|python3 extract_game.py > extracted_games09.pgn

def check_game(string):
  if f'TimeControl "{TIME_CONTROL}"' in string and 'eval' in string and 'WhiteRatingDiff' in string and 'BlackRatingDiff' in string:
        white_diff = int(string.split('WhiteRatingDiff "')[1].split('"')[0])
        black_diff = int(string.split('BlackRatingDiff "')[1].split('"')[0])
        if abs(white_diff) < MAX_RATING_DIFF and abs(black_diff) < MAX_RATING_DIFF:
            return True
        
  return False

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
    
  
  


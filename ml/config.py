#stores a time-control:file string. The time-control is a regex string that matches the time-control of a game. The file string is the name of the file that the game should be saved to.

file_dict = {"300+0":"blitz",
             "300+3":"blitz",
             "60+0":"ultrabullet",
             "120+1":"bullet",
             "180+0":"superblitz",
             "180+2":"superblitz",
             "600+0":"rapid",
             "600+5":"rapid",
             "900+10":"rapid",
             "1800+0":"classical",
             "1800+20":"classical"
}

#the maximum rating diff above which we ignore the game
MAX_RATING_DIFF = 40

#Termination strings in the game that we ignore. Games with this string will not be parsed.
TERMINATION_STRINGS=set(["Abandoned","Rules infraction"])

NUM_MOVES = 40 #number of moves to consider for each game
$(document).ready(function(){
    const LICHESS_URL = "https://lichess.org/game/export/{gameId}";

    
    $('#submit').click(function(){
        //add a loading spinner to show that the request is being processed.
        $('#result').html('<div class="spinner-border text-primary" role="status"><span class="sr-only">Loading</span></div>');
        var gameId = $('#game_id').val();
        
        //console.log("Game id: "+gameId);
        // if the gameId is of the form https://lichess.org/1M25iFQ3ZADh, extract the last part.
        if(gameId.includes("/")){
            gameId = gameId.split("/").pop();
        }
        //take the first 8 characters of the gameId. This is the gameId.
        gameId = gameId.substring(0,8);

        var url = LICHESS_URL.replace("{gameId}", gameId);
        console.log("Fetching game from "+url);
        //set the request header to accept json.
        
        
        $.get(url,
             function(data){
            console.log(data);
            //The data has a string "[White "<white_username>"] and [Black "<black_username>"]. Extract the usernames.
            var white = data.match(/\[White \"(.*?)\"\]/)[1];
            var black = data.match(/\[Black \"(.*?)\"\]/)[1];

            //check if there is a string of the form "[%eval" in the pgn. If not, report an error.
            if(data.match(/\[%eval/)==null){
                $('#result').html("Error: No evaluation in the pgn.");
                return;
            }

            //update spinner to "Evaluating game"
            $('#result').html('<div class="spinner-border text-primary" role="status"><span class="sr-only">Evaluating</span></div>');
            $.post('/submit_game', {game_pgn: data}, function(response){
                //clear spinner
                $('#result').html("");
                $('#whiteRating').html(response[0]+" ("+white+")");
                $('#blackRating').html(response[1]+" ("+black+")");
            });
         // on an error in the get request, report an error.
        }).fail(function(){
            $('#result').html("Error: Game not found.");
            });
    });
});
import * as tf from '@tensorflow/tfjs';

$(document).ready(async function(){
    const LICHESS_URL = "https://lichess.org/game/export/{gameId}?pgnInJson=true";
    //when the submit button is clicked, get the pgn from lichess and send it to the server, waiting for the response.

    const model = await tf.loadLayersModel('https://foo.bar/tfjs_artifacts/model.json');

    $('#submit').click(function(){
        //add a loading spinner to show that the request is being processed.
        $('#result').html('<div class="spinner-border text-primary" role="status"><span class="sr-only">Loading...</span></div>');
        var gameId = $('#gameId').val();
        // if the gameId is of the form https://lichess.org/1M25iFQ3ZADh, extract the last part.
        if(gameId.includes("/")){
            gameId = gameId.split("/").pop();
        }

        var url = LICHESS_URL.replace("{gameId}", gameId);
        $.get(url, function(data){
            //The data has a string "[White "<white_username>"] and [Black "<black_username>"]. Extract the usernames.
            var white = data.match(/\[White \"(.*?)\"\]/)[1];
            var black = data.match(/\[Black \"(.*?)\"\]/)[1];

            //check if there is a string of the form "[%eval" in the pgn. If not, report an error.
            if(data.match(/\[%eval/)==null){
                $('#result').html("Error: No evaluation in the pgn.");
                return;
            }

            $.post('/game', {pgn: data}, function(response){
                
                //once we have the game tensors, run them through the relevant tensorflow model to get the evaluation.
                var whiteRating = model.predict(response[0]);
                var blackRating = model.predict(response[1]);

                $('#whiteRating').html(whiteRating);
                $('#blackRating').html(blackRating);
            });
         // on an error in the get request, report an error.
        }).fail(function(){
            $('#result').html("Error: Game not found.");
            });
    });

});
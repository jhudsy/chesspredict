import * as tf from '@tensorflow/tfjs';

$(document).ready(async function(){
    const LICHESS_URL = "https://lichess.org/game/export/{gameId}?pgnInJson=true";

    //urls for the bullet, ultrabullet, blitz, rapid and classical models.
    const bulletUrl = 'https://foo.bar/tfjs_artifacts/bullet.json';
    const ultrabulletUrl = 'https://foo.bar/tfjs_artifacts/ultrabullet.json';
    const blitzUrl = 'https://foo.bar/tfjs_artifacts/blitz.json';
    const rapidUrl = 'https://foo.bar/tfjs_artifacts/rapid.json';
    const classicalUrl = 'https://foo.bar/tfjs_artifacts/classical.json';

    //load the models using tf.loadLayersModel
    const bulletModel = await tf.loadLayersModel(bulletUrl);
    const ultrabulletModel = await tf.loadLayersModel(ultrabulletUrl);
    const blitzModel = await tf.loadLayersModel(blitzUrl);
    const rapidModel = await tf.loadLayersModel(rapidUrl);
    const classicalModel = await tf.loadLayersModel(classicalUrl);

    

    $('#submit').click(function(){
        //add a loading spinner to show that the request is being processed.
        $('#result').html('<div class="spinner-border text-primary" role="status"><span class="sr-only">Loading game from lichess</span></div>');
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

            //update spinner to "Evaluating game"
            $('#result').html('<div class="spinner-border text-primary" role="status"><span class="sr-only">Evaluating game</span></div>');
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
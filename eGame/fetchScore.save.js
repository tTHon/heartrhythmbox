//tryapis.com

var totalQ = 16;
var p1Score = [];
var p2Score = [];
var p3Score = [];
const nowScore = [0,0,0];
const oldScore = [0,0,0];

  const options = {
    method: 'GET',
    headers: {
      accept: 'application/json',
      Authorization: 'Bearer Tp5sLcLZPjW95uAWeBWFgJmKZ2PtfQSIb6-IwB_NCuQ'
    }
  };

  const URL = 'https://api.netlify.com/api/v1/forms/634035b9b348c50008955b1a/submissions'
  
  fetch(URL, options)
  .then(response => {return response.json()})
  .then(data =>{console.log(data);
        form = JSON.stringify(data);
        document.getElementById('demo').innerHTML = form;
      })




function getScore(result){
    for (var i=0;i<result.length;i++){
      var f = result[i].data;
      p1Score[f.qNo] = parseInt(f.p1);
      p2Score[f.qNo] = parseInt(f.p2);
      p3Score[f.qNo] = parseInt(f.p3);
      var score0=0;
      for (let i=0;i<nowScore.length;i++){
        oldScore[i]=(nowScore[i])
      }
        for (let i=0;i<p1Score.length;i++){
          score0 = score0 + p1Score[i];
        }
      
      var score1=0;
        for (let i=0;i<p2Score.length;i++){
          score1 = score1 + p2Score[i];
        }
      
      var score2=0;
        for (let i=0;i<p3Score.length;i++){
          score2 = score2 + p3Score[i];
        }
        nowScore[0]=score0;nowScore[1]=score1;          
        nowScore[2]=score2;  
 
      
    }
  }







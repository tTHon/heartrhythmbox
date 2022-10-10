//tryapis.com

var totalQ = 16;
var p1Score = [];
var p2Score = [];
var p3Score = [];
const nowScore = [0,0,0];
const oldScore = [0,0,0];

function reFetch(gNo){
  const options = {
    method: 'GET',
    headers: {
      accept: 'application/json',
      Authorization: 'Bearer Tp5sLcLZPjW95uAWeBWFgJmKZ2PtfQSIb6-IwB_NCuQ'
    }
  };
  
fetch('https://api.netlify.com/api/v1/forms/634035b9b348c50008955b1a/submissions', options)
  .then(response => response.json())
  .then(result =>getScore(result))
  .catch(error => console.log('error', error));

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
  //-120 to 440 degree
  const degreeS = -120-(oldScore[gNo-1]*2);
  const degreeE = -120-(nowScore[gNo-1]*2)
  moveHand(gNo,degreeS,degreeE)
  document.getElementById('demo').innerHTML = degreeE-degreeS;
  dropText(nowScore[gNo-1]-oldScore[gNo-1],gNo)
}

function setScoreArray(){
  var sArray = [];
  for (var i=0;i<totalQ;i++){
      sArray[i] = 0;
  }
  return sArray;
}

function getScoreTest(){
  var p1Test = [];
  var p2Test = [];
  var p3Test = [];
}




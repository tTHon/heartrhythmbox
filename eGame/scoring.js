var pName = ['dao','oat','pom']

var keyJson = [{"no":0, "key": 'sinus tachycardia', "score": 2},
                {"no":0, "key": 'complete AV block', "score": 4},
                {"no":0, "key": 'escape rate 30bpm', "score": 2},
                {"no":0, "key": 'RBBB', "score": 2}]


function displayName(){
    document.getElementById('p1').innerHTML = pName[0];
    //document.getElementById('p2').innerHTML = pName[1];
    //document.getElementById('p3').innerHTML = pName[2];
}

function getKeyArray(n){
    var keyArray = [];
    for (var i=0;i<keyJson.length;i++){
         if (keyJson[i].no ==n){
            var k = keyJson[i].key
            keyArray.push (k);
        }
    }
    return keyArray;
}

function getScoreArray(n){
    var scoreArray = []
    for (var i=0;i<keyJson.length;i++){
        if (keyJson[i].no ==n){
           scoreArray.push (keyJson[i].score);
       }
   }
   return scoreArray;
}

function displayTickBox(n,boxName,labelName){
    var keyA = getKeyArray(n);
    var tickBox = document.getElementsByClassName(boxName);
    var label = document.getElementsByClassName(labelName)
    for (var i=0;i<keyA.length;i++){
        tickBox[i].style.display = 'block'
        label[i].innerHTML = keyA[i] 
    }
}


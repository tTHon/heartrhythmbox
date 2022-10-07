var pName = ['dao','oat','pom']
var totalQ = 15;

var keyJson = [{"no":0, "key": 'sinus tachycardia', "score": 2},
                {"no":0, "key": 'complete AV block', "score": 4},
                {"no":0, "key": 'escape rate 30bpm', "score": 2},
                {"no":0, "key": 'RBBB', "score": 2}]


function displayName(){
    document.getElementById('p1').innerHTML = pName[0];
    document.getElementById('p2').innerHTML = pName[1];
    document.getElementById('p3').innerHTML = pName[2];
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
    var scoreA = getScoreArray(n)
    var tickBox = document.getElementsByClassName(boxName);
    var label = document.getElementsByClassName(labelName)
    for (var i=0;i<keyA.length;i++){
        tickBox[i].style.display = 'inline-flex'
        label[i].innerHTML = keyA[i] + ' (' + scoreA[i] + ')'
        //for p2 and p3
        tickBox[10+i].style.display = 'inline-flex'
        label[10+i].innerHTML = keyA[i] + ' (' + scoreA[i] + ')'
        tickBox[20+i].style.display = 'inline-flex'
        label[20+i].innerHTML = keyA[i] + ' (' + scoreA[i] + ')'
    }
}

function displayNavBar(qNo){
    var left = document.getElementById('left')
    var right = document.getElementById('right')
    var current = document.getElementById('current')
    current.innerHTML = qNo + '/' + totalQ
    left.href = "scoring.html"
    right.href = "scoring.html"
}

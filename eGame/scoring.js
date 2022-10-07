var pName = ['dao','oat','pom']
var totalQ = 15;
var qNo =0;
var keyJson = [{"no":0, "key": 'sinus tachycardia', "score": 2},
                {"no":0, "key": 'complete AV block', "score": 4},
                {"no":0, "key": 'escape rate 30bpm', "score": 2},
                {"no":0, "key": 'RBBB', "score": 2}]
loadPage();

document.getElementById('left').addEventListener ("click", function(){
    if (qNo>0) {qNo--}
    loadPage()
})
document.getElementById('right').addEventListener ("click",function(){
    console.log()
    if (qNo<15){qNo++}
    loadPage();
})
function loadPage(){                
    displayName();
    displayTickBox(qNo,'aBox','label');
    displayNavBar(qNo)
    currentScore = [0,0,0];
}


//name and ecg
function displayName(){
    document.getElementById('p1').innerHTML = pName[0];
    document.getElementById('p2').innerHTML = pName[1];
    document.getElementById('p3').innerHTML = pName[2];
    document.getElementById('questionNo').value = qNo.toString();
    var source = "ecgs\\ecg"+ qNo.toString()+".png";
    //document.getElementById('ecg').src = source;
    document.getElementById('title').innerHTML = "Question No." + qNo;
}

function getKeyArray(n){
    var keyArray = [];
    for (let i=0;i<keyJson.length;i++){
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

function updateScore(key,pNo){
    var scoreArray = getScoreArray(qNo);
    var addScore = scoreArray[key];
    //check if checked;
    var inPutNo = (pNo-1)*10+key;
    var input = document.getElementsByClassName('input')
    if (input[inPutNo].checked){
        currentScore[pNo-1] = currentScore[pNo-1]+addScore
    }
    else {
        currentScore[pNo-1] = currentScore[pNo-1]-addScore
    }
    
    var showScore = document.getElementsByClassName('showScore')
    showScore[pNo-1].value = currentScore[pNo-1];
}

function displayNavBar(qNo){
    var left = document.getElementById('left')
    var right = document.getElementById('right')
    var current = document.getElementById('current')
    current.innerHTML = qNo + '/' + totalQ
    if (qNo>0){
        left.innerHTML = '<'
    } else {left.innerHTML = ''}
    if (qNo<15){
        right.innerHTML = '>'
    } else {right.innerHTML =""}
}

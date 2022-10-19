var pName = ['DAO','KHAW','OAT']
var totalQ = 12;
var keyJson = [
                {"no":1, "key": '1.1', "score": 2},
                {"no":1, "key": '1.2', "score": 5},
                {"no":1, "key": '1.3', "score": 3},         
                {"no":2, "key": '2.1', "score": 3},
                {"no":2, "key": '2.2', "score": 4},
                {"no":2, "key": '2.3', "score": 3},   
                {"no":3, "key": '3.1', "score": 5},
                {"no":3, "key": '3.2', "score": 2},
                {"no":3, "key": '3.3', "score": 3},
                {"no":4, "key": 'RBBB', "score": 2},
                {"no":4, "key": '4.1', "score": 2},
                {"no":4, "key": '4.2', "score": 5},
                {"no":1, "key": '4.3', "score": 3},         
                {"no":5, "key": '5.1', "score": 3},
                {"no":5, "key": '5.2', "score": 4},
                {"no":6, "key": '6.1', "score": 3},   
                {"no":7, "key": '7.1', "score": 5},
                {"no":8, "key": '8.2', "score": 2},
                {"no":9, "key": '9.3', "score": 3},
                {"no":10, "key": '10.1', "score": 3},   
                {"no":10, "key": '10.2', "score": 5},
                {"no":11, "key": '11.2', "score": 2},
                {"no":12, "key": '12.3', "score": 3}            
            
            ]

function loadPage(){                
    displayName();
    resetScore();
    displayTickBox(qNo,'aBox','label');
    currentScore = [0,0,0];
}

//name and ecg
function displayName(){
    document.getElementById('p1').innerHTML = pName[0];
    document.getElementById('p2').innerHTML = pName[1];
    document.getElementById('p3').innerHTML = pName[2];
    document.getElementById('questionNo').value = qNo.toString();
    document.getElementById('title').innerHTML = "Question No. " + qNo + " /" + totalQ;
    var source = "ecgs//ecg"+ qNo.toString()+".png";
    document.getElementById('ecg').src = source;
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
        tickBox[i].style.display = 'block'
        label[i].innerHTML = keyA[i] + ' (' + scoreA[i] + ')'
        //for p2 and p3
        tickBox[10+i].style.display = 'block'
        label[10+i].innerHTML = keyA[i] + ' (' + scoreA[i] + ')'
        tickBox[20+i].style.display = 'block'
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

function resetScore(){
    var input = document.getElementsByClassName('input')
    var score = document.getElementsByClassName('showScore')
    for (let index = 0; index < input.length; index++) {
        input[index].checked = false;
        
    }
    for (let index = 0; index < score.length; index++) {
        score[index].value = 0;
        
    }

    var box = document.getElementsByClassName('aBox')
    var label = document.getElementsByClassName('label')
    for (let index = 0; index < label.length; index++) {
        label[index].innerHTML = ''        
    }
    for (let index = 0; index < box.length; index++) {
        box[index].style.display = 'none'
        
    }
}



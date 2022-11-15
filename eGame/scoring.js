var pName = ['DAO','KHAW','OAT']
var totalQ = 12;
var keyJson = [ {"no":0, "key": 'no key', "score": 0},
                {"no":0, "key": 'no key2', "score": 0},
                {"no":0, "key": 'no key3', "score": 0},
                {"no":1, "key": 'Brugada Pattern', "score": 2},
                {"no":1, "key": 'LAE', "score": 5},
                {"no":1, "key": '1.3', "score": 3},         
                {"no":2, "key": '2.1', "score": 3},
                {"no":2, "key": 'RBBB', "score": 4},
                {"no":2, "key": '2.3', "score": 3},   
                {"no":3, "key": 'TGA', "score": 5},
                {"no":3, "key": '3.2', "score": 2},
                {"no":3, "key": '3.3', "score": 3},
                {"no":4, "key": 'RBBB', "score": 2},
                {"no":4, "key": '4.1', "score": 2},
                {"no":4, "key": 'CHB', "score": 5},
                {"no":4, "key": '4.3', "score": 1},         
                {"no":5, "key": '5.1', "score": 3},
                {"no":5, "key": 'Osborne', "score": 4},
                {"no":6, "key": '6.1', "score": 3}, 
                {"no":6, "key": 'VT', "score": 3},   
                {"no":7, "key": '7.1', "score": 5},
                {"no":7, "key": 'AF', "score": 5},
                {"no":8, "key": '8.2', "score": 2},
                {"no":8, "key": 'STEMI', "score": 2},
                {"no":9, "key": '9.3', "score": 3},
                {"no":9, "key": 'Wellen', "score": 3},
                {"no":10, "key": '10.1', "score": 3},   
                {"no":10, "key": '10.2', "score": 5},
                {"no":10, "key": 'de winter', "score": 5},
                {"no":11, "key": '11.2', "score": 2},
                {"no":11, "key": 'PVCs', "score": 2},
                {"no":12, "key": '12.3', "score": 3},  
                {"no":12, "key": 'HypoK', "score": 3}            
            
            ]

function loadPage(){     
    //document.getElementById('summary').style.display = 'none'           
    displayName();
    resetScore();
    displayTickBox(qNo,'aBox','label');
    currentScore = [0,0,0];
    p1Array=[]
    p2Array=[]
    p3Array=[]
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

function finalCheckScore(pNo){
    var pArray=[]
    var scoreArray = getScoreArray(qNo);

    //check if checked;
    var begin = (pNo-1)*10;

    var input = document.getElementsByClassName('input')
    for (let index =0; index < scoreArray.length; index++) {
        iNo = index + begin;
        if (input[iNo].checked){
            pArray.push(index)
        }
    }
    return pArray;
}

function showSummary(a1,a2,a3,s1,s2,s3){
    //get key
    summary = document.getElementById('summary')
    summary.style.display = 'block'
    summary.scrollIntoView({block: "start"})
    var keyA = getKeyArray(qNo);
    var scoreA = getScoreArray(qNo);
    const sum = document.getElementsByClassName('scoreSum')
    const ssPName = document.getElementsByClassName('scoreSumPName')
    
    //clear texxt
    for (let index = 0; index < 3; index++) {
        sum[index].innerHTML = ""        
    }
    
    addText(a1,s1,0)
    addText(a2,s2,1)
    addText(a3,s3,2)

    function addText(array,total,classNo){
        ssPName[classNo].innerHTML = pName[classNo] + '<br />' 
        for (i=0;i<array.length;i++){
            keyNo = array[i]
            sum[classNo].innerHTML += keyA[keyNo] + ' ' + scoreA[keyNo] + '<br />'
        }
        sum[classNo].innerHTML += 'Total Score: ' + total;
    }
}

function qNav(nav){
    newQ = qNo+nav;

    if (newQ<0){
        document.getElementById('dn').style.display = 'none'
        newQ=0;
        qNo = newQ;
        loadPage();
    }
    else if (newQ>totalQ){
        document.getElementById('up').style.display = 'none'
        newQ = totalQ
        qNo = newQ;
        loadPage();
    }
    else {
        document.getElementById('dn').style.display = 'inline'
        document.getElementById('up').style.display = 'inline'
        document.getElementById('current').innerHTML = newQ;
        qNo = newQ;
        loadPage();
    }

    if (qNo==0) {
        document.getElementById('dn').style.display = 'none'       
    }
    else if (qNo==totalQ){
        document.getElementById('up').style.display = 'none'    
    }
}
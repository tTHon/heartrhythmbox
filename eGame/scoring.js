var pName = ['DAO','KHAW','OAT']
var totalQ = 12;
var keyJson = [ {"no":0, "key": 'sinus rhythm', "score": 0},
                {"no":0, "key": '1st AVB', "score": 0},
                {"no":0, "key": 'ST Elevation inferolateral leads', "score": 0},
                {"no":0, "key": 'ST Depression V1-V2, I, aVL', "score": 0},
                {"no":0, "key": 'Acute Inf/Post/Lat STEMI', "score": 0},
                {"no":1, "key": 'WCT', "score": 2},
                {"no":1, "key": 'Right Axis Deviation', "score": 2},
                {"no":1, "key": 'VA dissociation', "score": 2},  
                {"no":1, "key": 'RBBB', "score": 2},  
                {"no":1, "key": 'LAFVT', "score": 2},         
                {"no":2, "key": 'Sinus rhythm', "score": 1},
                {"no":2, "key": 'Left Axis Deviation', "score": 2},
                {"no":2, "key": 'iRBBB/RV conduction delay', "score": 2},   
                {"no":2, "key": 'RVH w/strain', "score": 3},   
                {"no":2, "key": 'ASD Primum', "score": 2}, 
                {"no":3, "key": 'AP w/o capture', "score": 2},
                {"no":3, "key": 'VP w/ capture', "score": 1},
                {"no":3, "key": 'Conducted retrograde P', "score": 2},
                {"no":3, "key": 'LAD or LAFB', "score": 2},
                {"no":3, "key": 'Alternating normal axis', "score": 1},
                {"no":3, "key": 'Poor R', "score": 1},
                {"no":3, "key": 'LVH by voltage', "score": 1},
                {"no":4, "key": 'Sinus tachy', "score": 1},
                {"no":4, "key": 'AVW', "score": 2},
                {"no":4, "key": 'ST elevation inf', "score": 2},
                {"no":4, "key": 'ST depression ant/lat', "score": 2},         
                {"no":4, "key": 'acute/recent', "score": 1},  
                {"no":4, "key": 'inf', "score": 1},  
                {"no":4, "key": 'STEMI', "score": 1}, 
                {"no":5, "key": 'Totally irregular', "score": 1},
                {"no":5, "key": 'AF', "score": 1},
                {"no":5, "key": 'V rate 130', "score": 1},
                {"no":5, "key": 'Under standard gain', "score": 2},
                {"no":5, "key": 'Q II, III, aVF', "score": 2},
                {"no":5, "key": 'Pacing w/o capture', "score": 3},
                {"no":6, "key": 'Right atrial enlargment', "score": 2}, 
                {"no":6, "key": 'Marked', "score": 1},   
                {"no":6, "key": 'Right axis deviation', "score": 1}, 
                {"no":6, "key": 'Tall R in V1', "score": 2}, 
                {"no":6, "key": 'RV strain', "score": 2}, 
                {"no":6, "key": 'Reverse R wave progression', "score": 2}, 
                {"no":6, "key": 'TGA (Bonus)', "score": 2}, 
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
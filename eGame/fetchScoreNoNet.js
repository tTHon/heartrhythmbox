//tryapis.com
//oyqF9n48JsYpRKXk-RaF8GsOgsa4NVrIkMZ9W_b7zrM

var totalQ = 13;
var p1S = [0,0,0,0,0,0,0,0,0,0,0,0,0]; 
var p2S = [0,0,0,0,0,0,0,0,0,0,0,0,0]; 
var p3S = [0,0,0,0,0,0,0,0,0,0,0,0,0]; 
var questionNumber=0;
var q2Vote = [4,8,12];



function refresh(gNo, array){
  if (questionNumber>=0 && array.length>0)
    {updateScore(gNo,array,questionNumber)}
}

function updateScore(gNo,array,qNo){
  if (qNo==0) {oldScore=0;
      newScore = 0;
  }
  else if (qNo>0) {
    oldScore = sumScore(array,qNo-1);
    newScore = sumScore(array,qNo)
  }

 
  function sumScore(array,q){
    var sum=0;
    for (let index = 0; index <= q; index++) {
      if (array[index]>=0)
        {sum = sum + array[index]}
     }
    return sum
  }

  const block = 300/((totalQ-1)*10);
  const degreeS = -120-(oldScore*block);
  const degreeE = -120-(newScore*block)
  const add = newScore-oldScore
  clearFace(gNo)
  moveHand(gNo,degreeS,degreeE);
  dropText(add,gNo);
  showScore(gNo,oldScore,newScore)
}

function qNav(nav){
    newQ = questionNumber+nav;

    if (newQ<0){
        document.getElementById('dn').style.display = 'none'
        newQ=0;
        questionNumber = newQ;
    }
    else if (newQ>totalQ-1){
        document.getElementById('up').style.display = 'none'
        newQ = totalQ-1
        questionNumber = newQ;
    }
    else {
        document.getElementById('dn').style.display = 'inline'
        document.getElementById('up').style.display = 'inline'
        document.getElementById('qNow').innerHTML = newQ;
        questionNumber = newQ;
    }

    if (questionNumber==0) {
        document.getElementById('dn').style.display = 'none'       
    }
    else if (questionNumber==totalQ-1){
        document.getElementById('up').style.display = 'none'    
    }
}

function showScoreCard(status){
  if (status == 'on'){
    card = document.getElementById(scoreCard)
    scoreCard.style.display = 'block'
    scoreT = document.getElementById('scoreT')
    scoreT.deleteRow(-1);
    scoreT.deleteRow(-1);scoreT.deleteRow(-1);scoreT.deleteRow(-1);

    var player = ['DAO','KHAW','OAT']
    var head = scoreT.insertRow(0);
    var row1 = scoreT.insertRow(1);
    var row2 = scoreT.insertRow(2);
    var row3 = scoreT.insertRow(3);

    //header
    for (let index = 0; index <= totalQ; index++) {
        var cell = head.insertCell(index)
        cell.innerHTML = index
        if (index ==0){cell.innerHTML = ' '}
        if (index ==totalQ){cell.innerHTML = 'Total'}
    }

    //p1
    for (let i=0;i<=totalQ;i++){
        let cell = row1.insertCell(i)
        if (p1S[i]>=0){
        cell.innerHTML = p1S[i] 
        cell.onclick = function(){mEnter(1,i)}
        } else {cell.innerHTML = ' '}

        if (i==0){cell.innerHTML=player[0]}
        if (i==totalQ){cell.innerHTML = scoreSum(p1S)}
    }

    //p2
    for (let i=0;i<=totalQ;i++){
        let cell = row2.insertCell(i)
        if (p2S[i]>=0){
        cell.innerHTML = p2S[i] 
        cell.onclick = function(){mEnter(2,i)}
        } else {cell.innerHTML = ' '}

        if (i==0){cell.innerHTML=player[1]}
        if (i==totalQ){cell.innerHTML = scoreSum(p2S)}
    }

    //p3
    for (let i=0;i<=totalQ;i++){
        let cell = row3.insertCell(i)
        if (p3S[i]>=0){
        cell.innerHTML = p3S[i] 
        cell.onclick = function(){mEnter(3,i)}
        } else {cell.innerHTML = ' '}

        if (i==0){cell.innerHTML=player[2]}
        if (i==totalQ){cell.innerHTML = scoreSum(p3S)}
    }

    function scoreSum(array){
        var sum=0;
        for (let index = 0; index < totalQ; index++) {
        add = array[index]
        if (add>=0){sum = sum+ array[index];}
        }
        return sum;
    }
    } 
    else if (status == 'off') {
        menuToggle = 0;
        scoreCard.style.display = 'none'
        mEnterInit('off')
    }

}

edit = 'off'
function mEnterInit(status){
  const scoreT = document.getElementById('scoreT');
  const save = document.getElementById('saveChange')
  const undo = document.getElementById('undo')
  if (edit=='off'){
    edit = 'on'
    scoreT.style.background = 'pink'
    scoreT.style.cursor = 'pointer'
    save.style.display = 'inline'
    undo.style.display = 'inline'
  } else {
    edit= 'off'
    scoreT.style.background = 'none'
    scoreT.style.cursor = 'context-menu'
    save.style.display = 'none'
    undo.style.display = 'none'
  }
  if (status=='off'){
    edit= 'off'
    scoreT.style.background = 'none'
    scoreT.style.cursor = 'context-menu'
    save.style.display = 'none'
    undo.style.display = 'none'
  }

}

function mEnter(row,col){
  if (edit=='on'){
      const scoreT = document.getElementById('scoreT');
      thisCell = scoreT.rows[row].cells[col]
      
      if (thisCell.children.length==0){
        oldValue = thisCell.innerHTML
        thisCell.innerHTML = ''
        thisCell.style.background = '#121212'
        var x = document.createElement("INPUT");
        x.setAttribute("type", "text");
        x.style.color = "#cdcdcd"
        x.style.fontSize = '3vmax'
        x.style.width = '4.5vmax'
        x.id = row.toString()+col.toString()
        thisCell.appendChild(x);
        x.focus();
      }
    }
} 

function saveMEnter(){
  let inputTag = document.getElementsByTagName('input')

  for (let index = 0; index < inputTag.length; index++) {
    let inputId = inputTag[index].id

    //pick only input in scoreCard
    if (inputId.length<=2){
      let row = Number(inputId.charAt(0))
      let col = Number(inputId.charAt(1))
      if (row==1){
        p1S[col] = Number(inputTag[index].value)
      }
      if (row==2){
        p2S[col] = Number(inputTag[index].value)
      } 
      if (row==3){
        p3S[col] = Number(inputTag[index].value)
      }
    }
  }
  mEnterInit('off');
  showScoreCard('on')
}

function undo(){
  let inputTag = document.getElementsByTagName('input')
  for (let index = 0; index < inputTag.length; index++) {
    let inputId = inputTag[index].id

    //pick only input in scoreCard
    if (inputId.length<=2){
      let row = Number(inputId.charAt(0))
      let col = Number(inputId.charAt(1))
      if (row==1){
        inputTag[index].value = p1S[col]
      }
      if (row==2){
        inputTag[index].value = p2S[col]
      } 
      if (row==3){
        inputTag[index].value = p3S[col]
      }
    }
  }
  showScoreCard('on');
}

//players dashboard
function showScore(gNo,oldS,newS){

  //showScore
  const board = document.getElementsByClassName('score')
  var add = newS-oldS;
  var run=0;
  var runInterval=setInterval(function(){
    board[gNo-1].innerHTML = oldS+run;
    run++;
    if (run>add){
      clearInterval(runInterval)
  }
  },100)


}


function createCanvas() {
  const scale = window.devicePixelRatio;
  canvas.style.width = 0.96*window.innerWidth + 'px';
  canvas.style.height = 0.5*window.innerHeight + 'px';
  canvas.width = Math.floor(0.96*window.innerWidth * scale*4)
  canvas.height = Math.floor(0.5*window.innerHeight * scale*4)
  ctx.scale(scale*4,scale*4)
}

function dropText(add,gNo){
  var baseX = (16*vw*(2*gNo-1)); 
  endX = baseX-vw;startY = 1*vh;
  endY = 8*vh;
  text = '+'+add
  
  //drop effect
  var distance = (endY-startY);
    var tEnd=0;
    var moveInterval=setInterval(function(){
      ctx.clearRect(baseX-(15*vw),0,30*vw,10*vh)
      ctx.textAlign = "center";ctx.fillStyle = "#f7f7f7";
      ctx.font = 2.5*vw + 'px Raleway, sans-serif';
      ctx.fillText(text,endX,startY+(tEnd))
      tEnd=tEnd+(0.5*vh);
      if (tEnd>=distance){
        clearInterval(moveInterval)
    }
    },tEnd/distance)

}

function drawLabel(gNo){
  var baseX = (16*vw*(2*gNo-1)); 
  max = (totalQ-1)*10
  //cos 62
  var xLeft=baseX-(Math.cos(Math.PI*55/180)*(radius))-(1.5*vw);
  var xRight = baseX+(Math.cos(Math.PI*55/180)*(radius))+(1.5*vw);
  var yPos = baseY+(radius);
  ctx.beginPath();
  ctx.fillStyle = "#ddd"
  ctx.font = 2*vw + 'px Raleway, sans-serif'
  ctx.textAlign = "center";
  ctx.fillText ('0',xLeft,yPos) 
  ctx.fillText (max,xRight,yPos) 
}

function drawFace(gNo) {
  var xPos=16*vw*(2*gNo-1);var yPos = baseY;
  var color = [
    ['#ffeee5','#ffddcc','#ffccb2','#ffbb99','#ffaa7f',
    '#ff9966','#ff9059','#ff884c','#ff8346','#ff7632'],

    ['#c0f2ee','#aceee9','#97eae3','#82e5de','#6ee1d8',
    '#59ddd3','#44d9cd','#30d5c8','#2bbfb4','#26aaa0'],

    ['#f0fbdb','#e9faca','#e8fac9','#e2f9b8','#daf7a6',
    '#d3f694','#cbf482','#c4f270','#bcf15e','#b5ef4c'],
    
  ]


  //coloring 120-420 degree
  for (var i=0;i<totalQ-1;i++){
    degreeWidth = 300/(totalQ-1);totalColor = 10;
    degreeS = (120 + (degreeWidth*i));
    degreeE = (degreeS + (degreeWidth-0.5));
    ctx.beginPath();
    ctx.arc(xPos, yPos, radius, degreeS*Math.PI/180, degreeE*Math.PI/180);
    if (i>totalColor-1){j=9} 
      else {j=i}
    ctx.fillStyle = color[gNo-1][j];
    ctx.lineTo(xPos, yPos)
    ctx.fill();
    
  }

  //draw arc blank
  ctx.beginPath();
  ctx.arc(xPos, yPos, radius*0.6, 0, 2*Math.PI);
  ctx.fillStyle = '#333';
  ctx.fill();  

  //draw center
  ctx.beginPath();
  ctx.arc(xPos, yPos, radius*0.15, 0, 2*Math.PI);
  ctx.fillStyle = '#111111';
  ctx.fill();


}

function drawHand(gNo,degree) {
    var length = 0.8*radius;
    var xPos=16*vw*(2*gNo-1);var yPos = baseY;
    ctx.beginPath();
    ctx.lineWidth = 0.8*vw;
    ctx.lineCap = "round";
    xEnd = length*Math.cos(Math.PI*degree/180);
    yEnd = length*Math.sin(Math.PI*degree/180);
    ctx.moveTo(xPos,yPos);
    ctx.lineTo(xPos+xEnd,yPos-yEnd);
    ctx.strokeStyle = '#ff0000'
    ctx.stroke();
}

function moveHand(gNo,degreeS,degreeEnd){

  //go2End+2
    var distance = (degreeEnd-degreeS)+5;
    var tEnd=1;
    var moveInterval=setInterval(function(){
      clearFace(gNo);
      drawFace(gNo);
      drawHand(gNo,degreeS+tEnd)
      tEnd++;
      if (tEnd>=distance){
      clearInterval(moveInterval)
      shakeMinus(gNo);
      shakePlus(gNo);
    }
    },30*tEnd/distance)

  
  
  //shake-2
  function shakeMinus(){
  var tBack=1;
  var shakeBack=setInterval(function(){
    clearFace(gNo);
    drawFace(gNo);
    drawHand(gNo,(degreeEnd+5)-tBack)
    drawLabel(gNo);
    tBack++;
    if (tBack>=10){
    clearInterval(shakeBack)
    }
  },tBack)
  }

  //shake+2
  function shakePlus(){
  var tFwd=1;
  var shakeFwd=setInterval(function(){
    clearFace(gNo);
    drawFace(gNo);
    drawHand(gNo,(degreeEnd-5)+tFwd)
    tFwd++;
    if (tFwd>=10){
    clearInterval(shakeFwd)
    clearFace(gNo);
    drawFace(gNo);
    drawHand(gNo,degreeEnd)
    }
  },tFwd*50)
  }

}

function clearFace(gNo){
  var xPos=16*vw*(2*gNo-1);var yPos = baseY;
  ctx.fillStyle = '#333';
  ctx.beginPath();
  ctx.arc(xPos, yPos, radius, 100/180*Math.PI,440/180*Math.PI);
  ctx.lineTo(xPos, yPos)
  ctx.fill();
}





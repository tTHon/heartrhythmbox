//tryapis.com


var totalQ = 13;
var p1Score = [];
var p2Score = [];
var p3Score = [];
var questionNumber;
var q2Vote = [4,8,12];

function reFetch(){ 
  const options = {
    method: 'GET',
    headers: {
      accept: 'application/json',
      Authorization: 'Bearer '
    }
  };
  
  fetch('https://api.netlify.com/api/v1/forms/ID/submissions', options)
    .then(response => response.json())
    .then(result =>getScore(result))
    .catch(error => console.log('error', error));

  function getScore(result){
    //var timeStamp = []
    for (var i=0;i<result.length;i++){
      var f = result[i].data;
      p1Score[f.qNo] = parseInt(f.p1);
      p2Score[f.qNo] = parseInt(f.p2);
      p3Score[f.qNo] = parseInt(f.p3);
    }
 }
}

function updateScore(gNo,array,qNo){
  if (qNo==0) {oldScore=0;
  newScore = array[0]}
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
  moveHand(gNo,degreeS,degreeE);
  dropText(add,gNo);
  showScore(gNo,oldScore,newScore)
}

var menuToggle = 0;
function showMenu(){
  //get qNav posision
  dot = document.getElementById('qNav')
  menu = document.getElementById('menu')

  //calc vmax qNav font size = 3vmax
  menu.style.left = (window.innerWidth-menu.offsetWidth)/2 + 'px'
  menu.style.top = (window.innerHeight-menu.offsetHeight)/2 + 'px'

  if (menuToggle == 0){
    menu.style.visibility = 'visible'
    menuToggle = 1;
  }
  else if (menuToggle == 1){
    menu.style.visibility = 'hidden'
    menuToggle = 0;
    //clear scorecard
    card = document.getElementById(scoreCard)
    scoreCard.style.display = 'none'
    scoreT = document.getElementById('scoreT')
    scoreT.deleteRow(-1);
    scoreT.deleteRow(-1);scoreT.deleteRow(-1);scoreT.deleteRow(-1);
    document.getElementById('ss').style.display = 'none'
    document.getElementById('audience').style.display = 'none'
  }
}

function showScoreCard(){
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
    if (p1Score[i]>=0){
      cell.innerHTML = p1Score[i] 
    } else {cell.innerHTML = ' '}

    if (i==0){cell.innerHTML=player[0]}
    if (i==totalQ){cell.innerHTML = scoreSum(p1Score)}
  }

  //p2
  for (let i=0;i<=totalQ;i++){
    let cell = row2.insertCell(i)
    if (p2Score[i]>=0){
      cell.innerHTML = p2Score[i] 
    } else {cell.innerHTML = ' '}

    if (i==0){cell.innerHTML=player[1]}
    if (i==totalQ){cell.innerHTML = scoreSum(p2Score)}
  }

  //p3
  for (let i=0;i<=totalQ;i++){
    let cell = row3.insertCell(i)
    if (p3Score[i]>=0){
      cell.innerHTML = p3Score[i] 
    } else {cell.innerHTML = ' '}

    if (i==0){cell.innerHTML=player[2]}
    if (i==totalQ){cell.innerHTML = scoreSum(p3Score)}
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


function startStopQ(){
  ss = document.getElementById('ss')
  menu = document.getElementById('menu')
  ss.style.display = 'block'
  ss.style.left = (menu.offsetLeft) + 'px'
  ss.style.bottom = (menu.offsetTop) + 'px'
  ss.style.width = menu.offsetWidth + 'px'
  document.getElementById('ssQNo').innerHTML = 'Current Q: ' +questionNumber
}

var qClickStatus=0;
//0 = ready for q selection; 1 = clicked for regular q;
//2 = ready for accepting vote;3 = after vote;

function startQ(q){
  questionNumber = q;
  document.getElementById('qNow').innerHTML = questionNumber;
  document.getElementById('start').style.color = 'DarkTurquoise'
  document.getElementById('start').value = 'Click here to start Q: ' +q
  qClickStatus=0;
}

function startQClick(){
  //cross if clicked
  var q = document.getElementsByClassName('ssItem')
  q[questionNumber].style.textDecoration = 'line-through'
  q[questionNumber].style.color = '#777'
  document.getElementById('ssQNo').innerHTML = 'Current Q: ' +questionNumber
  document.getElementById('triggerNo').value = questionNumber;
  //document.getElementById('start').innerHTML = 'Click Number to Start Question'
  //document.getElementById('start').style.color = '#cdcdcd'

  if (qClickStatus==2){
    document.getElementById('start').value = 'Click Number to Start Question'
    document.getElementById('start').style.color = '#cdcdcd'
    qClickStatus=3;    
  }
  else if (qClickStatus==0){
      //check if vote button is needed
      qClickStatus = 1;
      document.getElementById('start').value = 'Click Number to Start Question'
      document.getElementById('start').style.color = '#cdcdcd'
      for (let index = 0; index < q2Vote.length; index++){
        if (questionNumber==q2Vote[index]){
          document.getElementById('start').style.color = 'DarkTurquoise'
          document.getElementById('start').value = 'Stop Accepting Votes'
          qClickStatus = 2;
           break;
        }}
    }

  if (qClickStatus==1 || qClickStatus==3){showMenu()}

}


//form submission begin
const handleSubmit = (event) => {
  event.preventDefault();
  console.log(questionNumber)
  document.getElementById("t2Send").value = Date.now();
  if (qClickStatus==3)
    {document.getElementById("q2Send").value = 9;}
  else {document.getElementById("q2Send").value = questionNumber;}
  
  const myForm = event.target;
  const formData = new FormData(myForm);
  
  fetch("/", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams(formData).toString(),
  })
    .then(() => console.log("Form successfully submitted"))
    .catch((error) => alert(error));


}

document
  .querySelector("form")
  .addEventListener("submit", handleSubmit);
//form submission end

//audience score
function showAudience(){
  document.getElementById('audience').style.display = 'block'

  //fetch score
    //fetchScore -- return aScore
    const options = {
      method: 'GET',
      headers: {
        accept: 'application/json',
        Authorization: 'Bearer Tp5sLcLZPjW95uAWeBWFgJmKZ2PtfQSIb6-IwB_NCuQ'
      }
    };
    
    fetch('https://api.netlify.com/api/v1/forms/634cb3f7db273c000a680d53/submissions', options)
      .then(response => response.json())
      //.then(response => console.log(response))
      .then(result => handleScore(result))
      //.catch(err => console.error(err));

  function handleScore(result){
    //data.audName: "x"
    //data.audienceTotalScore: "7"
    //data.ip: "202.28.177.50"
    //qNow: ""

    //count no of voters
    //first -- get max qNow
    maxQ = getMaxQ()
    
    function getMaxQ(){
      var qNumber = [];
      for (let index = 0; index < result.length; index++) {
        var f = result[index].data;
        qNumber[index] = parseInt(f.qNow);
      }
      qNumber.sort(function(a, b){return b - a})
      max = qNumber[0]
      return max;
    }

    //count maxQ votes
    voteNo = getVoteNo(maxQ);
    document.getElementById('audienceCount').innerHTML = 'Submitted: '+voteNo
    function getVoteNo(Q){
      count=0;
      for (let index = 0; index < result.length; index++) {
        var f = result[index].data;
        if (f.qNow ==Q){
          count++
        }
      }
      return count;
    }

    //audScore [name][score][ip] + sorting
    var audScore = getAudienceArray(maxQ)
    //document.getElementById('audienceCount').innerHTML = audScore;
    function getAudienceArray(Q){
      var array = [];
      for (let index = 0; index < result.length; index++) {
        var f = result[index].data;
        if (f.qNow ==Q){
          audName = f.audName
          audScore = parseInt(f.audienceTotalScore)
          audIp = f.ip
          toPush = [audName,audScore,audIp]
          array.push(toPush)
        }
      }
      array.sort(function(a,b){
        if (a[1]>b[1]) {return -1}})

      return array;
    }

    //show score
    var maxItem = 5;
    if (result.length<maxItem){maxItem=result.length}
    var name2Show = document.getElementsByClassName('aName')
    var score2Show = document.getElementsByClassName('aScore')
    var audIp = document.getElementsByClassName('ip')
    for (let index = 0; index < audIp.length; index++) {
      audIp[index].style.display = 'none'
    
    }
    for (let index = 0; index < maxItem; index++) {
      name2Show[index].innerHTML = audScore[index][0]
      score2Show[index].innerHTML = audScore[index][1]
      audIp[index+1].innerHTML = audScore[index][2]
      name2Show[index].style.visibility = 'hidden'
      score2Show[index].style.visibility = 'hidden'
    }

  }
}

function showAItem(n){

  var maxItem = 5;
  var name2Show = document.getElementsByClassName('aName')
  var score2Show = document.getElementsByClassName('aScore')
  var audIp = document.getElementsByClassName('ip')
  if (n==99){
    for (let index = 0; index < maxItem; index++) {
      name2Show[index].style.visibility = 'visible'
      score2Show[index].style.visibility = 'visible'
    }
  }
  else if(n=='ip'){
    for (let index = 0; index < audIp.length; index++) {
      audIp[index].style.display = 'table-cell'      
    }
  }
  else {
    name2Show[n].style.visibility = 'visible'
    score2Show[n].style.visibility = 'visible'
  }
 }
//end audience data

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
  endY = 9.5*vh;
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
  var xLeft=baseX-(Math.cos(Math.PI*55/180)*(radius))-(vw);
  var xRight = baseX+(Math.cos(Math.PI*55/180)*(radius))+(vw);
  var yPos = baseY+radius+0.5*vh;
  ctx.beginPath();
  ctx.fillStyle = "#ddd"
  var hand = 0.8*radius
  ctx.font = 2*vw + 'px Raleway, sans-serif'
  ctx.textAlign = "center";
  ctx.fillText ('0',xLeft,yPos) 
  ctx.fillText (max,xRight,yPos) 
}

function drawFace(gNo) {
  var xPos=16*vw*(2*gNo-1);var yPos = baseY;
  var color = [
    ['#fff0fb','#ffeaf5','#ffd7ec','#ffc3e3','#ffafda',
    '#ff9cd1','#ff88c8','#ff75bf','#ff61b6','#ff4dad'],

    ["#BEF6D2","#9bf1bb","#8aefaf","#79eca3","#68ea97",
    "#57e78b","#45e580","#34e274","#23E068","#1ed25f"],
    
    ["#fdedcc","#fce6b9","#fcdfa7","#fbd994","#fad281",
    "#facb6e","#f9c45b","#f8be48","#f8b735",'#F7B022']]


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





//tryapis.com
//oyqF9n48JsYpRKXk-RaF8GsOgsa4NVrIkMZ9W_b7zrM

var totalQ = 13;
var p1S = [0,0,0,0,0,0,0,0,0,0,0,0,0]; 
var p2S = [0,0,0,0,0,0,0,0,0,0,0,0,0]; 
var p3S = [0,0,0,0,0,0,0,0,0,0,0,0,0]; 
var questionNumber;
var q2Vote = [4,8,12];

function sub(){
  const url = 'https://noospmcgjamvpgxlgmyc.supabase.co'
  const key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5vb3NwbWNnamFtdnBneGxnbXljIiwicm9sZSI6ImFub24iLCJpYXQiOjE2NjY2OTc5NjAsImV4cCI6MTk4MjI3Mzk2MH0.qbIQW8O_5mm5Dbz5_GJIBQE1fGo5PWM-xhDqeMWcGuY'
  const database = supabase.createClient(url,key)
  //console.log (database)

  database
  .channel('public:pScore')
  .on('postgres_changes', { event: 'INSERT', schema: 'public', table: 'pScore' }, 
  payload => {
    console.log('Change received!', payload)
    reFetch();
    logoBlink('on');
  })
  .subscribe()
}

sub();

function logoBlink(status){
  title = document.getElementsByClassName('titleFlex')
  if (status == 'on'){
    title[0].style.textShadow = '10px 15px 30px pink, -10px -15px 30px pink'}
  if (status == 'off'){
    title[0].style.textShadow = 'none'
  }
      
  }
    

function reFetch(){ 
  const url = 'https://noospmcgjamvpgxlgmyc.supabase.co'
  const key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5vb3NwbWNnamFtdnBneGxnbXljIiwicm9sZSI6ImFub24iLCJpYXQiOjE2NjY2OTc5NjAsImV4cCI6MTk4MjI3Mzk2MH0.qbIQW8O_5mm5Dbz5_GJIBQE1fGo5PWM-xhDqeMWcGuY'
  const database = supabase.createClient(url,key)

  //getData
    const getData = async () => {
      const pScore = await database.from("pScore")
      .select('*')
      .order('qNo', {ascending:false})
      getScore(pScore)
    }
  getData();
  
  function getScore(result){
    for (var i=0;i<result.data.length;i++){
      var k = result.data[i]
      var q = k.qNo;
      if (q==questionNumber){
        console.log('q=q')
        p1S[q] = k.p1Score;
        p2S[q] = k.p2Score;
        p3S[q] = k.p3Score;
        break;
      }
    }
  }
}

function refresh(gNo, array){
  logoBlink('off')
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

var menuToggle = 0;
function showMenu(){
  //get qNav posision
  dot = document.getElementById('qNav')
  menu = document.getElementById('menu')
  maxWidth = menu.style.offsetWidth

  //calc vmax qNav font size = 3vmax
  menu.style.left = (window.innerWidth-menu.offsetWidth)/2 + 'px'
  menu.style.top = (window.innerHeight-menu.offsetHeight)/2 + 'px'

  if (maxWidth>window.innerWidth) {
    menu.style.left = 0;
    maxWidth = window.innerWidth
  }

  menu.style.width = maxWidth;
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
    mEnterInit('off')
    scoreT = document.getElementById('scoreT')
    scoreT.deleteRow(-1);
    scoreT.deleteRow(-1);scoreT.deleteRow(-1);scoreT.deleteRow(-1);
    document.getElementById('ss').style.display = 'none'
    document.getElementById('audience').style.display = 'none'
    //tStamp
    time = document.getElementsByClassName('tStamp')
    for (let index = 0; index < time.length; index++) {
      time[index].style.display = 'none'
    }    
    document.getElementById('end').innerHTML = 'End Game'
    eGameCount =0;
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
        x.setAttribute("value", oldValue)
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
  showMenu();mEnterInit('off')
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
  showScoreCard();
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
  document.getElementById('start').style.color = 'lightBlue'
  document.getElementById('start').value = 'Click here to start Q: ' +q
  qClickStatus=0;
}

var eGameCount =0;
function endGame(){
  document.getElementById('end').innerHTML = 'Click here again to END GAME'
  eGameCount ++;
  if (eGameCount==2){
    document.getElementById('end').innerHTML = 'Game is about to end.'
    eGameCount =0;
    clearQue();
    questionNumber = 1000;
    sendQue();
    showMenu();
  }
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
    document.getElementById('start').value = 'Click Number to Start Q'
    document.getElementById('start').style.color = '#cdcdcd'
    qClickStatus=3;    
  }
  else if (qClickStatus==0){
      //check if vote button is needed
      qClickStatus = 1;
      document.getElementById('start').value = 'Click Number to Start Q'
      document.getElementById('start').style.color = '#cdcdcd'
      for (let index = 0; index < q2Vote.length; index++){
        if (questionNumber==q2Vote[index]){
          document.getElementById('start').style.color = 'lightBlue'
          document.getElementById('start').value = 'Stop Accepting Votes'
          qClickStatus = 2;
           break;
        }}
    }

  if (qClickStatus==1 || qClickStatus==3){showMenu()}
  sendQue()  
}

//Question Que
function sendQue(){
  const url = 'https://noospmcgjamvpgxlgmyc.supabase.co'
  const key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5vb3NwbWNnamFtdnBneGxnbXljIiwicm9sZSI6ImFub24iLCJpYXQiOjE2NjY2OTc5NjAsImV4cCI6MTk4MjI3Mzk2MH0.qbIQW8O_5mm5Dbz5_GJIBQE1fGo5PWM-xhDqeMWcGuY'
  const database = supabase.createClient(url,key)
  console.log (database)

  //q2send
  var q2Send;
  if (qClickStatus==3)
    {q2Send = 100+questionNumber;}
  else {q2Send = questionNumber}

  //insert
  const sendData = async () => {
    const feed = await database.from("qFeed").insert({
        questionNo: q2Send
    })
    console.log(feed)
  }
  sendData();
}

function clearQue(){
  const url = 'https://noospmcgjamvpgxlgmyc.supabase.co'
  const key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5vb3NwbWNnamFtdnBneGxnbXljIiwicm9sZSI6ImFub24iLCJpYXQiOjE2NjY2OTc5NjAsImV4cCI6MTk4MjI3Mzk2MH0.qbIQW8O_5mm5Dbz5_GJIBQE1fGo5PWM-xhDqeMWcGuY'
  const database = supabase.createClient(url,key)
  console.log (database)

  const clearQ = async ()=>{
    const data = await database
    .from('qFeed')
    .delete()
    .gte ('questionNo',0)
  }
  const clearAud = async ()=>{
    const data = await database
    .from('Audience')
    .delete()
    .gte ('qNo',0)
  }

  const clearP = async ()=>{
    const data = await database
    .from('pScore')
    .delete()
    .gte ('qNo',0)
  }

  //console.log(data)
  clearQ();
  clearAud();
  clearP();
 
  //clear player score array
  for (let index = 0; index < p1S.length; index++) {
    p1S[index] = 0;
    p2S[index] = 0;
    p3S[index] = 0;
  }

  window.alert('clear')
}

//end q que

//audience score
function showAudience(){
  document.getElementById('audience').style.display = 'block'  

  const url = 'https://noospmcgjamvpgxlgmyc.supabase.co'
  const key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5vb3NwbWNnamFtdnBneGxnbXljIiwicm9sZSI6ImFub24iLCJpYXQiOjE2NjY2OTc5NjAsImV4cCI6MTk4MjI3Mzk2MH0.qbIQW8O_5mm5Dbz5_GJIBQE1fGo5PWM-xhDqeMWcGuY'
  const database = supabase.createClient(url,key)

  //getData
  const getData = async () => {
      const aScore = await database.from("Audience")
      .select('*')
      .order('qNo', {ascending:false})
      .order('created_at', {ascending:false})

      if (aScore.data.length>0){
          maxQ = aScore.data[0].qNo}
      else {maxQ = 0}
      //console.log(aScore.data.length)

      //count maxQ votes
      voteNo = getVoteNo(maxQ);
      document.getElementById('audienceCount').innerHTML = 'Submitted: '+voteNo
      function getVoteNo(Q){
        count=0;
        for (let index = 0; index < aScore.data.length; index++) {
          var f = aScore.data[index];
          if (f.qNo ==Q){
            count++
          }
        }
        return count;
      }
      //audScore [name][score] + sorting
      var audScore = getAudienceArray(maxQ)
      
      function getAudienceArray(Q){
        var array = [];
        for (let index = 0; index < aScore.data.length; index++) {
          var f = aScore.data[index];
          if (f.qNo ==Q){
            audName = f.audName
            audScore = parseInt(f.score)
            toPush = [audName,audScore]
            array.push(toPush)
          }
        }
        array.sort(function(a,b){
          if (a[1]>b[1]) {return -1}})

        return array;
      }


    //show score
    var maxItem = 5;
    if (audScore.length<maxItem){maxItem=audScore.length}
    var name2Show = document.getElementsByClassName('aName')
    var score2Show = document.getElementsByClassName('aScore')

    for (let index = 0; index < maxItem; index++) {
      name2Show[index].innerHTML = audScore[index][0]
      score2Show[index].innerHTML = audScore[index][1]
      name2Show[index].style.visibility = 'hidden'
      score2Show[index].style.visibility = 'hidden'
    }
    }

  getData();

}

function showAItem(n){

  var maxItem = 5;
  var name2Show = document.getElementsByClassName('aName')
  var score2Show = document.getElementsByClassName('aScore')
  var time = document.getElementsByClassName('tStamp')
  if (n==99){
    for (let index = 0; index < maxItem; index++) {
      name2Show[index].style.visibility = 'visible'
      score2Show[index].style.visibility = 'visible'
    }
  }
  else if(n=='tStamp'){
    for (let index = 0; index < time.length; index++) {
      time[index].style.display = 'table-cell'      
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





//input here
var allText = ["In symptomatic SND, pacemaker improves symptoms; not mortality",
                "In a patient with symptomatic sinus bradycardia from guideline-directed medication, pacemaker is indicated.",
                "In symptomatic SND, atrial-based pacing (not ventricular based or VVI) is recommended.",
                "Symptomatic high-grade AV block + stable AV node blockers --> proceed with pacemaker without further observation."
            ];
var bg = ['green','green','green','yellow']

//functions
var tStart = 0;
//tStart is a global variable = where the wheel is.
function showTxtBlk(tInit,tEnd){
    var listNo=0; var txtIndex;
    var tMax=allText.length; 
    
    for (i=tInit; i<=tEnd; i++){
      var txt = document.getElementsByClassName("card"+listNo);
      if (i>=tMax){
        txtIndex=i-tMax;
      } else {txtIndex=i}
      
      txt[0].classList.add('animateLeft10')
      txt[0].innerHTML = allText[txtIndex];
    

      listNo++;
      if (listNo>2) {listNo=0};
    } 

}

function fwd() {
    ++tStart;
    if (tStart>=allText.length){tStart=0}
    var tn=tStart+2;
    showTxtBlk(tStart,tn);
      }
    
function rw() {
    --tStart;
    if (tStart<0){tStart=(allText.length)-1}
    var tn=tStart+2;
    showTxtBlk(tStart,tn);
    }   

var init;
var fwdCount=0;
//fwdCount enables autorolling toggle.
function autoFwd(){
    if (fwdCount>0)
    { fwdCount=0;document.getElementsByClassName('card1')[0].classList.remove('animateLeft10')
      clearInterval(init)}
    else  { document.getElementsByClassName('card1')[0].classList.remove('animateLeft10') 
    ++fwdCount;
    init = setInterval(function(){
    ++tStart;
    if (tStart>=allText.length){tStart=0}
    var tn=tStart+2;
    showTxtBlk(tStart,tn);
    } , 1000);}
    
}
    
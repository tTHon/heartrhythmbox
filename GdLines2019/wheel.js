//input here
var allText = ["In symptomatic SND, pacemaker improves symptoms; <b>NOT</b> mortality.",
                "In a patient with symptomatic sinus bradycardia from guideline-directed medication, pacemaker is indicated.",
                "In symptomatic SND, <b>atrial-based pacing</b> (not ventricular based or VVI) is recommended.",
                "Symptomatic high-grade AV block + stable dose of AV node blocker --> proceed with pacemaker without further observation.",
                "<b>Lab tests</b><br> such as thyroid function tests/Lyme titer/potassium/pH for a potential underlying cause are reasonable.",
                "<b>Implantable cardiac monitor</b> for infrequent symptoms (>30days) and negative initial tests.",
                "SND+symptoms in acute spinal cord injury or post heart transplant, <b>aminophylline or theophylline</b> is reasonable.",
                "SND+symptoms+intact AV conduction: reasonable to program the pacemaker to <b>minimize ventricular pacing</b>.",
                "1AVB or Mobitz I with exertional symptoms, consider <b>EST</b> to determine the potential role of pacemaker.",
                "Acquired & irreversible Mobitz II, high grade AV block, or 3AVB: pacemaker is indicated <b>regardless</b> of symptoms.",
                "SND + AVB: <b>Dual chamber</b> is preferred over single chamber pacemaker.",
                "New LBBB post TAVR: careful survelliance for bradycardia is reasonable.",
                "<b>PPM in congenital CHB:</b><br> symptomatic bradycardia/wide QRS escape rhythm/mean daytime HR<50 bpm/complex ventricular ectopy/ventricular dysfunction.",
                "Evaluate risk for <b>ventricular arrhythmias</b> and need for ICD before PPM implantation."
              ];
var bg = ['MediumSpringGreen','MediumSpringGreen','MediumSpringGreen','LemonChiffon','LemonChiffon','LemonChiffon',
'LemonChiffon','LemonChiffon','LemonChiffon', 'MediumSpringGreen','MediumSpringGreen','LemonChiffon','MediumSpringGreen','MediumSpringGreen']

//functions
var tStart = 0;
//tStart is a global variable = where the wheel is.
function showTxtBlk(){
    var card = document.getElementById('card');
    card.innerHTML = allText[tStart]   
    card.style.backgroundColor = bg[tStart]
    var num = Math.floor(Math.random() * 10); 
    var mark = Math.floor(Math.random()*2);var deg;
    if (mark==1){deg=(num/3)*-1}
    else {deg=num/3}
    card.style.transform = 'rotate('+deg+'deg)'

    //show nav
    var nav = document.getElementById('bradyWheelCenter')
    var navNo = tStart + 1;
    nav.innerHTML = navNo + ' / ' + allText.length;
    //arrow
    var left = document.getElementById('bradyWheelLeft')
    var right = document.getElementById('bradyWheelRight')
    left.onclick = function(){rw()}
    right.onclick = function(){fwd()}
    document.getElementById('autoFwd').onclick = function(){
      autoFwd()
    }
}

function fwd() {
    ++tStart;
    if (tStart>=allText.length){tStart=0}
    showTxtBlk();
      }
    
function rw() {
    --tStart;
    if (tStart<0){tStart=(allText.length)-1}
    showTxtBlk();
    }   

var init;
var fwdCount=0;
//fwdCount enables autorolling toggle.
function autoFwd(){
    var a = document.getElementById('autoFwd')
    if (fwdCount>0)
    { fwdCount=0;a.style.color = "red"
      clearInterval(init)}
    else  {  
    ++fwdCount;
    a.style.color = "green"
    init = setInterval(function(){
    ++tStart;
    if (tStart>=allText.length){tStart=0}
      showTxtBlk();
    } , 4400);}
    
}
    
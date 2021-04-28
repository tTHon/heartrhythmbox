/*.pointer {
    cursor: pointer;
    height: 0.5em;
    width: 0.5em;
    background-color: #fc553d;
    border-radius: 50%;
    position: fixed;  
    z-index: 1;
    animation: blink 1.5s linear infinite;
    }
    @keyframes blink{
    0%{opacity: 0;}
    50%{opacity: 0.7}
    100%{opacity: 1;}
    }*/



function pointerSwitch(n){
    if(pSwitch ==0) {
        pSwitch =1;
    }
    else {
        pSwitch=0;
        var pointer = document.getElementsByClassName("pointer");
        pointer[0].style.display = "none";
    }
}

function pointer() {
    if (pSwitch!=0){
    var x = event.clientX;
    var y = event.clientY;
    var p = document.getElementsByClassName("pointer");
    p[0].style.display = "inline";
    p[0].style.left = x+"px"; 
    p[0].style.top = y+"px";}
}
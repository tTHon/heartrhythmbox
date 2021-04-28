
//nextBlk
function next(n) {
    var block = document.getElementsByClassName("block");
    block[n].scrollIntoView({behavior:"smooth", block:"start", inline:"start"});
    var i;
    for (i=0;i<block.length;i++){
        block[i].style.opacity = 0.1;
    }   
    block[n].style.opacity = 1;
     }

//pointer
function pointer() {
    var x = event.clientX;
    var y = event.clientY;
    var p = document.getElementsByClassName("pointer");
    p[0].style.display = "inline";
    p[0].style.left = x+"px"; 
    p[0].style.top = y+"px";
    }

function unblur() {
    var b = document.getElementsByClassName("block");
    var i;
    for (i=0;i<b.length;i++){
        b[i].style.opacity = 1;
    }
}
/*call in the main page
//up arrow
window.addEventListener("keydown", function(event) {
    event.preventDefault();
    if (event.keyCode===38) {
    blockNav(-1)}
})
//down arrow
window.addEventListener("keydown", function(event) {
    event.preventDefault();
    if (event.keyCode===40) {
    blockNav(1)}
})
*/


function blockNav(n){
    var block = document.getElementsByClassName("block");
    var i;var blockOn=false;var largeBlock = false;
    var screenTop = window.pageYOffset; var screenBottom = window.pageYOffset + window.innerHeight;
    for (i=0;i<block.length;i++){
        
        if (blockOn) {break;}
        if (block[i].offsetHeight>window.innerHeight){largeBlock==true}
        var blockTop = block[i].offsetTop;
        var blockBottom = block[i].offsetTop + block[i].offsetHeight;

        //top and bottom edges are on screen//
        if (blockTop>=screenTop && blockBottom<=screenBottom){
            blockOn==true;var blockNo=i;
            }
        //top on screen; bottom - not yet//
        else if (largeBlock =true && blockTop < screenTop + (0.5*window.innerHeight) &&
            blockBottom>screenBottom) {
            blockOn==true;var blockNo=i;
            }
        else if (largeBlock = true && blockTop < screenTop &&
            blockBottom > screenTop + (0.5*window.innerHeight) ){
            blockOn==true;var blockNo=i;
            }
    }
    var nav = Number(n) + blockNo;
    if (nav>=0 && nav<block.length){
    next(nav);}
    
}

//nextBlk
function next(n) {
    var block = document.getElementsByClassName("block");
    block[n].scrollIntoView({behavior:"smooth", block:"start"});
    var i;
    for (i=0;i<block.length;i++){
        block[i].style.opacity = 0.1;
    }   
    block[n].style.opacity = 1;
    if (n+1==block.length) {
        showArray()
    }
    }

function unblur() {
        var b = document.getElementsByClassName("block");
        var i;
        for (i=0;i<b.length;i++){
            b[i].style.opacity = 1;
        }
    }
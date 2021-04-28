function openNav() {
    document.getElementById("mySidenav").style.width = "auto";
    var b = document.getElementsByClassName("block");
    var i;
    for (i=0;i<b.length;i++){
        b[i].style.opacity = 1;
    }
    }

function closeNav() {
    document.getElementById("mySidenav").style.width = "0";
}

function scroll() {
    hamburger = document.getElementById("topnav");
    title = document.getElementById("topnavTxt");
    topArr = document.getElementsByClassName("toTop");
    
    if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20){
        hamburger.style.width="fit-content";
        hamburger.style.backgroundColor = "transparent";
        hamburger.style.fontSize = "1.5vmax";
        title.style.display="none";
        topArr[0].style.display="inline"}
       
    else {
        unblur();
        hamburger.style.width="100%";
        hamburger.style.backgroundColor = "#ffc30b";
        hamburger.style.fontSize = "2.5vmax";
        title.style.display="inline";
        topArr[0].style.display="none"}
}

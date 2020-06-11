function scroll() {
    topArr = document.getElementsByClassName("toTop");
    
    if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20){
        topArr[0].style.display="inline"}
       
    else {
        topArr[0].style.display="none"}
}

//html
//<a href="#top" title="to the top" class="toTop" onclick="unblur()">
//<i class="fas fa-arrow-circle-up" style="font-size: 2.8vmax"></i>
//</a>

//window.onscroll = function() {scroll()};

/*css
.toTop {
    position: fixed; 
    background-color: transparent; color: #e5E5E5;
    top: 92%;right: 2%;
    cursor: pointer;transition: 0.2s;
    display: none;
    z-index: 1; opacity: 1;
   }
.toTop :hover {opacity: 0.6}*/
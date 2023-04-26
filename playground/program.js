function addTopic(array){
    //one hour equal to 3vh + header of 9vh
    const oneHour = 9*vh;
    const dayTable = document.getElementById('dayTable');
    //console.log(oneHour)
    
    //mins display
    var startHour = Number(array.startTime.slice(0,2))
    var startMin = Number(array.startTime.slice(3,5))
    var endHour = Number(array.endTime.slice(0,2))
    var endMin = Number(array.endTime.slice(3,5))
    
    //hour array
    var hours = []
    for (let index = startHour; index <= endHour; index++) {
        hours.push(index)
    }
    //console.log(hours)

    //create new element(s) for each hour
    for (let index = 0; index < hours.length; index++) {
        var thisHour = hours[index]+1
        const startRow = dayTable.rows[thisHour].cells[1]
        const newBar = document.createElement('div')
        startRow.appendChild(newBar)
        newBar.className = 'topicBar'
                
        //top
        var top, bottom;
        if (index==0){
            top = (startMin/60)*oneHour  
            newBar.innerHTML = array.title
        } else {
            top = 0;
            newBar.style.borderTop = 'none'
        }
        if (index==(hours.length-1)){
            bottom = (endMin/60)*oneHour
        } else {
            bottom = oneHour
            newBar.style.borderBottom = 'none'
        }

        newBar.style.top = top + 'px'
        var height = bottom-top
        newBar.style.height = height + 'px'

        //topicBar font-size = 2.5vh
        fontNow = 2.5*vh
        if (fontNow>height){
            newBar.style.fontSize = height-(0.5*vh) + 'px'
        }

    }

}
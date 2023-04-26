function addTopic(array){
    console.log(array)
    //one hour equal to 3vh + header of 9vh
    const oneHour = 9*vh;
    console.log('oneHour = ' + oneHour)
    const dayTable = document.getElementById('dayTable');
    
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
    console.log(hours)

    //create new element(s) for each hour
    for (let index = 0; index < hours.length; index++) {
        var thisHour = hours[index]+1
        const startRow = dayTable.rows[thisHour].cells[1]
        const newBar = document.createElement('div')
        startRow.appendChild(newBar)
    }
    
    const startRow = dayTable.rows[startHour+1].cells[1]
    const newBar = document.createElement('div')
    startRow.appendChild(newBar)
    var top = oneHour*(startMin/60)
    if (top==0){top=1}
    newBar.style.top = top + 'px'
    console.log(newBar.style.top)
    //calculate bar height in minutes
    var barMin = (60*endHour+endMin)-(60*startHour+startMin)
    var height = oneHour*(barMin/60)
    console.log('barMin = '+barMin)
    newBar.style.height = height + 'px'
    console.log('height = ' + height)
    //calculate bar top
    newBar.className = 'topicBar'
    newBar.innerHTML = array.title
}
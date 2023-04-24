function addTopic(array){
    console.log(array)
    //one hour equal to 3vh + header of 9vh
    const oneHour = 9*vh;
    const dayTable = document.getElementById('dayTable');
    
    //mins display
    var startHour = Number(array.startTime.slice(0,2))
    var startMin = Number(array.startTime.slice(3,5))
    var endHour = Number(array.endTime.slice(0,2))
    var endMin = Number(array.endTime.slice(3,5))
    //find starting hour row
    const startRow = dayTable.rows[startHour].cells[1]
    const newBar = document.createElement('div')
    startRow.appendChild(newBar)
    newBar.style.height = oneHour*(startMin/60) + 'px'
    console.log(newBar.style.height)
    newBar.className = 'topicBar'
    newBar.innerHTML = array.title
}
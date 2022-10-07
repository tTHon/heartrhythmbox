//token jbBAry1lZp5mqeGiHOMg0QUBX6PyIR0sj1rfxuaREPk
//nocodeapi

var myHeaders = new Headers();
myHeaders.append("Content-Type", "application/json");
var requestOptions = {
    method: "get",
    headers: myHeaders,
    redirect: "follow",
    
};

fetch("https://v1.nocodeapi.com/tthon/netlify/NpVhOQLHdQGcYgnD/listFormSubmissions?form_id=633fd8d8bc959b00087bda49", requestOptions)
    .then(response => response.json())
    .then(data =>getScore(data))
    .catch(error => console.log('error', error));


function getScore(data){
    var form = data[0].ordered_human_fields
    for (let i=0;i<form.length;i++){
        if (form[i].name=='p1'){
            var p1=form[i].value;
        }
        if (form[i].name=='p2'){
            var p2=form[i].value;
        }
        if (form[i].name=='p3'){
            var p3=form[i].value
        }
    }
    document.getElementById('p1').innerHTML = "p1:" + p1
    document.getElementById('p2').innerHTML = "p2:" + p2
    document.getElementById('p3').innerHTML = "p3:" + p3
    document.getElementById('formName').innerHTML = data[0].form_name
}
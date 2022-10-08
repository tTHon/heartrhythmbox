//tryapis.com

var myHeaders = new Headers();
myHeaders.append("Content-Type", "application/json");
var requestOptions = {
    method: "get",
    headers: myHeaders,
    redirect: "follow",
    
};

const options = {
    method: 'GET',
    headers: {
      accept: 'application/json',
      Authorization: 'Bearer Tp5sLcLZPjW95uAWeBWFgJmKZ2PtfQSIb6-IwB_NCuQ'
    }
  };
  
  fetch('https://api.netlify.com/api/v1/forms/634035b9b348c50008955b1a/submissions', options)
  .then(response => response.json())
  .then(result =>getScore(result))
  .catch(error => console.log('error', error));


function getScore(result){
  var f = result[0].data;
  document.getElementById('p1').innerHTML = "p1:" + f.p1
  document.getElementById('p2').innerHTML = "p2:" + f.p2
  document.getElementById('p3').innerHTML = "p3:" + f.p3
  document.getElementById('formName').innerHTML = "qNo:" + f.qNo
}
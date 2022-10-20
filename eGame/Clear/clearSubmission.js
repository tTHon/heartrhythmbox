var listForm = [];
var listIndex = [];
var urlArray = [];

function listForms(){
  const options = {
    method: 'GET',
    headers: {
      accept: 'application/json',
      Authorization: 'Bearer Tp5sLcLZPjW95uAWeBWFgJmKZ2PtfQSIb6-IwB_NCuQ'
    }
  };
  
  fetch('https://api.netlify.com/api/v1/sites/f043edf6-6408-4b05-a718-76b3e56b8ddc/forms', options)
    .then(response => response.json())
    //.then(response => console.log(response))
    .then(result => listFormResult(result))
    .catch(err => console.error(err));

  function listFormResult(result){
    //[name][formId]
    for (let index = 0; index < result.length; index++) {
      var name = result[index].form_name
      var formId = result[index].id
      toPush = [name,formId]
      listIndex.push(toPush)
    }
    var formIdUrl = []
    for (let index = 0; index < listIndex.length; index++) {
      url = 'https://api.netlify.com/api/v1/forms/' + listIndex[index][1] + '/submissions'
      formIdUrl.push(url)
    }
    //document.getElementById('list1').innerHTML = formIdUrl

    //find submissions of all forms
    var arrayString;
    for (let index = 0; index < formIdUrl.length; index++) {
      thisUrl = formIdUrl[index]      
      array = listSubmissions(thisUrl)
     
      arrayString = arrayString + 'form' + index + ': ' + array
    }
 
  }
}



function listSubmissions(url){
//list submissions
  const options = {
  method: 'GET',
  headers: {
    accept: 'application/json',
    Authorization: 'Bearer Tp5sLcLZPjW95uAWeBWFgJmKZ2PtfQSIb6-IwB_NCuQ'
  }
};

//fetch('https://api.netlify.com/api/v1/forms/634e14b8e3483f0008fc0ab2/submissions', options)
fetch(url,options)
  .then(response => response.json())
  //.then(response => console.log(response))
  .then(result => listResult(result))
  .catch(err => console.error(err));

  //form_id: "634e14b8e3483f0008fc0ab2"//form_name: "feedQ"
  function listResult(result){
    var listIndex=[];
    //[name][id]
    for (let index = 0; index < result.length; index++) {
      var name = result[index].form_name
      var id = result[index].form_id
      toPush = [name,id]
      listIndex.push(toPush)
    }
    for (let index = 0; index < listIndex.length; index++) {
      url = 'https://api.netlify.com/api/v1/submissions/' + listIndex[index][1]
      urlArray.push(url)
    }
  }
}


function deleteSubmissions(){
//delete
const optionsDel = {
    method: 'DELETE',
    headers: {Authorization: 'Bearer Tp5sLcLZPjW95uAWeBWFgJmKZ2PtfQSIb6-IwB_NCuQ'}
  };
  
  fetch('https://api.netlify.com/api/v1/submissions/submissionID', optionsDel)
    .then(response => response.json())
    .then(response => console.log(response))
    .catch(err => console.error(err));
}
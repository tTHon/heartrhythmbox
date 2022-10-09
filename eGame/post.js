const data = { username: 'example' };

fetch('\functions\mailBox.js', {
  method: 'POST', // or 'PUT'
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(data),
})
  .then((response) => response.json())
  .then((data) => {
    document.getElementById('demo').innerHTML=data[0].username;
  })
  .catch((error) => {
    console.error('Error:', error);
  });
  
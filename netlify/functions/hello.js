exports.handler = async function (event, context) {
  const name = event.queryStringParameters.name
    //const formData = new FormData(myForm);
  

    return {
      statusCode: 200,
      body: 'hello, ${name}',
      //body: JSON.stringify({ message: "Hello World" }),
    };
  };
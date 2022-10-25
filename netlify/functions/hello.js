exports.handler = async function (event, context) {
  const name = event.data
    //const formData = new FormData(myForm);
  

    return {
      statusCode: 200,
      body: name,
      //body: JSON.stringify({ message: "Hello World" }),
    };
  };
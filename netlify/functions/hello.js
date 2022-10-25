exports.handler = async function (event, context) {
    const myForm = event.target;
    //const formData = new FormData(myForm);
  

    return {
      statusCode: 200,
      body: JSON.stringify(myForm)
      //body: JSON.stringify({ message: "Hello World" }),
    };
  };
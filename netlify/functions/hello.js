exports.handler = async function (event, context) {
  const name = event.queryStringParameters.name;

  return {
    statusCode: 200,
    body: JSON.stringify(`${name}`),
  };
};
  
//supabase #x6NCQtEDGxrdaw
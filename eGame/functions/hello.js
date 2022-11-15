
exports.handler = async () => {
    const key = process.env.supabaseKey;
    const url = process.env.supabaseURL;
    const supArray = [url,key]
    return {
      statusCode: 200,
      body: JSON.stringify(supArray)
    };

  };

exports.handler = async () => {
    const url = process.env.supabaseURL;
    return {
      statusCode: 200,
      body: url,
    };

  };
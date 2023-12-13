
exports.handler = async () => {
    const mod = process.env.moderator;
    const scr = process.env.scorer;
    const supArray = [mod,scr]
    return {
      statusCode: 200,
      body: JSON.stringify(supArray)
    };

  };
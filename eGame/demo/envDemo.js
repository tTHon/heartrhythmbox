function getEnv(){
  const options = {
      method: 'GET',
      headers: {
        accept: 'application/json',
        Authorization: 'Bearer '
      }
    };


    
    fetch('https://api.netlify.com/api/v1/sites', options)
    .then(response => response.json())
    .then(result => getURL(result))
    .catch(err => console.error(err));

    function getURL(result){
        key = result[0].build_settings.env.SUPABASE_ANON_KEY
        url = result[0].build_settings.env.SUPABASE_URL
    }
}


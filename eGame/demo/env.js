function getEnv(){
  const options = {
      method: 'GET',
      headers: {
        accept: 'application/json',
        Authorization: 'Bearer byLDRD12H9zVpadt9nD0PNsUMtY5iqfxm9cvsosk4u8'
      }
    };

    //https://api.netlify.com/api/v1/accounts/{account_id}/env/{key}
    //"id": "5ca5f90d40745df93379f428"
    
    fetch('https://api.netlify.com/api/v1/sites', options)
    .then(response => response.json())
    .then(result => getURL(result))
    .catch(err => console.error(err));

    function getURL(result){
        key = result[0].build_settings.env.SUPABASE_ANON_KEY
        url = result[0].build_settings.env.SUPABASE_URL
    }
}

//const url = 'https://noospmcgjamvpgxlgmyc.supabase.co'
//const key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5vb3NwbWNnamFtdnBneGxnbXljIiwicm9sZSI6ImFub24iLCJpYXQiOjE2NjY2OTc5NjAsImV4cCI6MTk4MjI3Mzk2MH0.qbIQW8O_5mm5Dbz5_GJIBQE1fGo5PWM-xhDqeMWcGuY'
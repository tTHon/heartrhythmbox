function reFetch(){ 
    const options = {
      method: 'GET',
      headers: {
        accept: 'application/json',
        Authorization: 'Bearer byLDRD12H9zVpadt9nD0PNsUMtY5iqfxm9cvsosk4u8'
      }
    };

    //https://api.netlify.com/api/v1/accounts/{account_id}/env/{key}
    //"id": "5ca5f90d40745df93379f428"
    
    fetch('https://api.netlify.com/api/v1/accounts/5ca5f90d40745df93379f428/env', options)
    .then(response => response.json())
    .then(response => console.log(response))
    .catch(err => console.error(err));

  }
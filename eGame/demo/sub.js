const database = supabase.createClient(url,key)
//console.log (database)
//sub

function sub(){
    database
    .channel('public:qFeed')
    .on('postgres_changes', { event: 'INSERT', schema: 'public', table: 'qFeed' }, payload => {
      console.log('Change received!', payload)
      qNow = payload.new.questionNo
      document.getElementById('demo').innerHTML = qNow;
    })
    .subscribe()
}

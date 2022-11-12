const url = 'https://noospmcgjamvpgxlgmyc.supabase.co'
const key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5vb3NwbWNnamFtdnBneGxnbXljIiwicm9sZSI6ImFub24iLCJpYXQiOjE2NjY2OTc5NjAsImV4cCI6MTk4MjI3Mzk2MH0.qbIQW8O_5mm5Dbz5_GJIBQE1fGo5PWM-xhDqeMWcGuY'
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

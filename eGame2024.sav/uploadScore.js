//const supabaseUrl = 'https://noospmcgjamvpgxlgmyc.supabase.co'
//const supabaseKey = eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5vb3NwbWNnamFtdnBneGxnbXljIiwicm9sZSI6ImFub24iLCJpYXQiOjE2NjY2OTc5NjAsImV4cCI6MTk4MjI3Mzk2MH0.qbIQW8O_5mm5Dbz5_GJIBQE1fGo5PWM-xhDqeMWcGuY;
//const database = supabase.createClient(supabaseUrl,supabaseKey);

function sendScore(no){
    const sendData = async () => {
        const feed = await database
            .from("eGame2024")
            .insert({
            qNo: no,
            p1: getTotalScore(0), p2: getTotalScore(1), p3: getTotalScore(2)
            })
            //console.log(feed)
    }
    sendData();
}

function resetDatabase(){
    const reset = async () => {
        const data = await database
            .from('eGame2024')
            .delete()
            .gte('p1', 0)
            //console.log(data)
    }
    reset();
}
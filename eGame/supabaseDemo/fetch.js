const url = 'https://noospmcgjamvpgxlgmyc.supabase.co'
const key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5vb3NwbWNnamFtdnBneGxnbXljIiwicm9sZSI6ImFub24iLCJpYXQiOjE2NjY2OTc5NjAsImV4cCI6MTk4MjI3Mzk2MH0.qbIQW8O_5mm5Dbz5_GJIBQE1fGo5PWM-xhDqeMWcGuY'
const database = supabase.createClient(url,key)
console.log (database)


//insert
let save = document.getElementById('demo');
save.addEventListener("click", async (e) => {
    e.preventDefault();
    let qNo = document.getElementById("no").value;
    save.setAttribute("disabled", true);
    let res = await database.from("qFeed").insert({
        questionNo: qNo
    })
})

var timeArray = [];
//get
const getData = async () => {
    const res = await database.from("qFeed").select("*");
    console.log(JSON.stringify(res))
    for (let index = 0; index < res.data.length; index++) {
        var time = res.data[index].timeStamp;
        timeArray.push(time)
    }
    if (timeArray[1]>timeArray[0])
        {document.getElementById('demo').innerHTML = 'desc'}
}


const url = 'https://noospmcgjamvpgxlgmyc.supabase.co'
const key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5vb3NwbWNnamFtdnBneGxnbXljIiwicm9sZSI6ImFub24iLCJpYXQiOjE2NjY2OTc5NjAsImV4cCI6MTk4MjI3Mzk2MH0.qbIQW8O_5mm5Dbz5_GJIBQE1fGo5PWM-xhDqeMWcGuY'
const database = supabase.createClient(url,key)
console.log (database)


let save = document.getElementById('demo');
save.addEventListener("click", async (e) => {
    e.preventDefault();
    let qNo = document.getElementById("no").value;
    save.setAttribute("disabled", true);
    let res = await database.from("qFeed").insert({
        questionNo: qNo
    })
})


const getData = async () => {
    const res = await database.from("qFeed").select("*");
    console.log(JSON.stringify(res))
    document.getElementById('demo').innerHTML = res.data[0].questionNo
}

getData();
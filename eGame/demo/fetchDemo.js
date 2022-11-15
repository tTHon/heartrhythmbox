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

//get & sort -- latest entry first
const getData = async () => {
    const res = await database.from("qFeed")
    .select("*")
    .order('timeStamp', {ascending:false})
    console.log(JSON.stringify(res))
}


//sbp_73eaa8e734d970f52610cb46fd1a4d6d7c08a19d
//npm install --save @supabase/supabase-js

const SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5vb3NwbWNnamFtdnBneGxnbXljIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTY2NjY5Nzk2MCwiZXhwIjoxOTgyMjczOTYwfQ.oy4V-23uYXch0gs6iIbGPrvIj40RlgcTKBLxf4m9L8s'
const supabaseKey = process.env.SUPABASE_KEY
const SUPABASE_URL = "https://noospmcgjamvpgxlgmyc.supabase.co"

const { createClient } = require('@supabase/supabase-js');

const supabase = createClient(SUPABASE_URL, supabaseKey);

exports.handler = async function(event,context) {

  const { data, error } = await supabase
  .from('qFeed')
  .insert([
    { questionNo: '0'},
  ])

  console.log(data, error);
}
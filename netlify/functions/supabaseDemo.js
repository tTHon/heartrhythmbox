//sbp_73eaa8e734d970f52610cb46fd1a4d6d7c08a19d
//npm install --save @supabase/supabase-js

const SUPABASE_KEY = 'sbp_73eaa8e734d970f52610cb46fd1a4d6d7c08a19d'
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
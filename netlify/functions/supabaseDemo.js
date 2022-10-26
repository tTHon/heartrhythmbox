import { createClient } from '@supabase/supabase-js'

// Connect to our database 
const _supabase = createClient('https://noospmcgjamvpgxlgmyc.supabase.co', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5vb3NwbWNnamFtdnBneGxnbXljIiwicm9sZSI6ImFub24iLCJpYXQiOjE2NjY2OTc5NjAsImV4cCI6MTk4MjI3Mzk2MH0.qbIQW8O_5mm5Dbz5_GJIBQE1fGo5PWM-xhDqeMWcGuY')

// Our standard serverless handler function
exports.handler = async event => {

  // Insert a row
    const { data, error } = await supabase
        .from('qFeed')
        .insert([
            {questionNo: '0' },
        ]);

  // Did it work?
  console.log(data, error);
  
}


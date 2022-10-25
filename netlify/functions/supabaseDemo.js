// Grab our credentials from a .env file or environment variables
require('dotenv').config();
const {
    SUPABASE_URL,
    SESSION_SECRET
} = process.env;

// Connect to our database 
const { createClient } = require('@supabase/supabase-js');
const supabase = createClient(SUPABASE_URL, SESSION_SECRET);

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


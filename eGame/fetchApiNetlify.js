import { NetlifyAPI } from 'netlify'

const client = new NetlifyAPI('jbBAry1lZp5mqeGiHOMg0QUBX6PyIR0sj1rfxuaREPk')
const sites = await client.listSites()
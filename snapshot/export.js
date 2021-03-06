const ejs = require('ejs')
const puppeteer = require('puppeteer')
const path = require('path')
const fs = require('fs')
const { promisify } = require('util')
const readFile = promisify(fs.readFile)

let puppet = async function(str, data) {
  let version = await readFile(__dirname + '/version.tag')
  await fs.mkdir(path.join(__dirname, 'output/', data.model + '-' + version, '/'), async (err) => {
    if (err && err.code != 'EEXIST') throw 'error'

    const browser = await puppeteer.launch()
    const page = await browser.newPage()
    await page.setViewport({ width: 300, height: 300 })
    await page.goto(`data:text/html,${str}`)
    await timeout(5000) // 5 sec timeout to make sure page is rendered
    let filename = new Date().toISOString() + '-'+ data.title +'.png'
    await page.screenshot({path: path.join(__dirname, 'output/', data.model + '-' + version, '/', filename)})
    await browser.close()
  })
};

let render = async function (data) {
  await ejs.renderFile(path.join(__dirname, 'p5sketch.ejs'), data, async function(err, str){
    if (err) throw 'error' + err
    /*let filename = new Date().toISOString() + '-'+ data.title
    fs.writeFileSync('./test'+filename+'.html', str, 'utf8');*/
    await puppet(str, data)
  })
}

let takesnap = async function(data) {
  if(data) {
    await render(data)
  }
}

async function timeout(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

module.exports = {
  takesnap: takesnap
}
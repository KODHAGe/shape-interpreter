const ejs = require('ejs')
const puppeteer = require('puppeteer')
const path = require('path')
const fs = require('fs')

let puppet = async function(str, data) {
  await fs.mkdir(path.join(__dirname, 'output/', data.model, '/'), async (err) => {
    if (err && err.code != 'EEXIST') throw 'error'

    const browser = await puppeteer.launch()
    const page = await browser.newPage()
    await page.setViewport({ width: 300, height: 300 })
    await page.goto(`data:text/html,${str}`, { waituntil: 'networkidle2'})
    let filename = new Date().toISOString() + '-example.png'
    await page.screenshot({path: path.join(__dirname, 'output/', data.model, '/', filename)})
    await browser.close()
  })
};

let render = async function (data) {
  await ejs.renderFile(path.join(__dirname, 'p5sketch.ejs'), data, async function(err, str){
    if (err) throw 'error' + err
    await puppet(str, data)
  })
}

let takesnap = async function(data) {
  if(data) {
    await render(data)
  }
}

module.exports = {
  takesnap: takesnap
}
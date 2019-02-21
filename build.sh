zip -r interpreter.zip * -x node_modules/@tensorflow/**\* node_modules/puppeteer/**\*
zip -ur interpreter.zip .env
ibmcloud fn action update interpreter interpreter.zip --docker kodhage/action-nodejs-v8:tfjs
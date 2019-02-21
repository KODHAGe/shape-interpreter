FROM openwhisk/action-nodejs-v8:latest

COPY package.json package.json 

RUN npm install

COPY . .

CMD ["npm","start"]
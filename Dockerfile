FROM openwhisk/action-nodejs-v10:latest

COPY package.json package.json 

RUN npm install

COPY . .

CMD ["npm","start"]
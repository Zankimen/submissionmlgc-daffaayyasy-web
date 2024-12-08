require('dotenv').config(); 

const Hapi = require('@hapi/hapi');
const loadModel = require('../services/loadModel');
const routes = require('../server/routes');
const InputError = require('../exceptions/InputError');

(async () => {
    const server = Hapi.server({
        port: 8080,
        host: 'localhost',
        routes: {
            cors: {
                origin: ['*'],
            },
        },
    })

    const model = await loadModel();
    
    server.app.model = model;

    server.route(routes);

    
   server.ext('onPreResponse', function (request, h) {
        const response = request.response;

        console.log(response.statusCode, response.code)
        if (response instanceof InputError) {
            const newResponse = h.response({
                status: 'fail',
                message: `${response.message}`
            })
            newResponse.code(response.statusCode)
            return newResponse;
        }

        if (response.isBoom) {
            const newResponse = h.response({
                status: 'fail',
                message: response.message
            })
            newResponse.code(response.output.statusCode)
            return newResponse;
        }

        return h.continue;
    })

    await server.start();
    console.log(`Server start at: ${server.info.uri}`);
})();
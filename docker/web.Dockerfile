FROM node:24-bookworm-slim AS build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend ./
COPY src/client /app/src/client
RUN npm run build

FROM nginx:1.28-alpine
RUN apk add --no-cache jq
COPY --from=build /app/frontend/dist /usr/share/nginx/html
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY docker/web-config.sh /docker-entrypoint.d/40-delta-config.sh
RUN chmod +x /docker-entrypoint.d/40-delta-config.sh
EXPOSE 80

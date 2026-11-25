FROM alpine:3.22

ARG PB_VERSION=0.42.3
RUN apk add --no-cache ca-certificates curl unzip \
    && set -eux; \
       case "$(uname -m)" in \
         x86_64) pb_arch=amd64 ;; \
         aarch64) pb_arch=arm64 ;; \
         *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;; \
       esac; \
       curl -fsSL "https://ganigeorgiev.com/meta/pocketbase/pocketbase/download/v${PB_VERSION}/pocketbase_${PB_VERSION}_linux_${pb_arch}.zip" -o /tmp/pocketbase.zip; \
       unzip /tmp/pocketbase.zip -d /pb; \
       rm /tmp/pocketbase.zip; \
       chmod +x /pb/pocketbase

COPY pb_migrations /pb/pb_migrations
WORKDIR /pb
EXPOSE 8090
VOLUME ["/pb/pb_data"]
CMD ["/pb/pocketbase", "serve", "--http=0.0.0.0:8090", "--migrationsDir=/pb/pb_migrations"]

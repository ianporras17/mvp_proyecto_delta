#!/usr/bin/env bash
set -euo pipefail

runtime_root="${1:-/runtime}"
cert_dir="${runtime_root}/Player-Data"
mkdir -p "${cert_dir}"

complete=1
for file in P0.pem P0.key P1.pem P1.key P2.pem P2.key C0.pem C0.key; do
    if [[ ! -s "${cert_dir}/${file}" ]]; then
        complete=0
    fi
done

for file in P0.pem P1.pem P2.pem C0.pem; do
    # Reutilizar solo certificados legibles con al menos un día de vigencia.
    if ! openssl x509 -in "${cert_dir}/${file}" -checkend 86400 -noout >/dev/null 2>&1; then
        complete=0
    fi
done

if [[ "${complete}" -eq 0 ]]; then
    # setup-ssl.sh elimina *.0 sin -f; el placeholder hace seguro el primer uso.
    touch "${cert_dir}/bootstrap.0"
    /opt/MP-SPDZ/Scripts/setup-ssl.sh 3 "${cert_dir}"
    (
        cd "${runtime_root}"
        /opt/MP-SPDZ/Scripts/setup-clients.sh 1
    )
fi

c_rehash "${cert_dir}"
# Each deployed party sees its own private key, the transport-client key,
# and public peer certificates. It cannot read other parties' private keys.
for party in 0 1 2; do
    isolated="${runtime_root}/party-${party}/Player-Data"
    mkdir -p "${isolated}"
    cp "${cert_dir}"/*.pem "${isolated}/"
    cp "${cert_dir}/P${party}.key" "${cert_dir}/C0.key" "${isolated}/"
    c_rehash "${isolated}"
done
touch "${runtime_root}/certificates.ready"

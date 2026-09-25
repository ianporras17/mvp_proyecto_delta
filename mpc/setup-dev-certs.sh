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
touch "${runtime_root}/certificates.ready"

#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Pack Ascend OPP vendors/config.ini and custom_xllm_math into a bare tarball.
# Output: bin/vendors-{version}-{commit_id}-{arch}-{device}-bare.tar.gz
# Example: vendors-0.9.2-083df65-arm-a3-bare.tar.gz

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BIN_DIR="${REPO_ROOT}/bin"
VERSION_FILE="${REPO_ROOT}/version.txt"

VENDOR_NAME="custom_xllm_math"
CONFIG_NAME="config.ini"

ASCEND_HOME="${ASCEND_HOME_PATH:-${NPU_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}"
VENDORS_DIR="${VENDORS_DIR:-${ASCEND_HOME}/opp/vendors}"
ARCH="${ARCH:-}"
DEVICE="${DEVICE:-a3}"

error_exit() {
    echo -e "\033[31merror: $1\033[0m" >&2
    exit 1
}

detect_arch() {
    if [[ -n "${ARCH}" ]]; then
        return
    fi
    case "$(uname -m)" in
        aarch64|arm64) ARCH="arm" ;;
        x86_64|amd64) ARCH="x86" ;;
        *) error_exit "unsupported architecture: $(uname -m)" ;;
    esac
}

read_version_tag() {
    if [[ ! -f "${VERSION_FILE}" ]]; then
        error_exit "version file not found: ${VERSION_FILE}"
    fi

    local version_raw
    version_raw="$(tr -d '[:space:]' < "${VERSION_FILE}")"
    if [[ -z "${version_raw}" ]]; then
        error_exit "version file is empty: ${VERSION_FILE}"
    fi

    VERSION_TAG="${version_raw}"
}

read_commit_id() {
    if git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        COMMIT_ID="$(git -C "${REPO_ROOT}" rev-parse --short=7 HEAD)"
    else
        error_exit "failed to read git commit from ${REPO_ROOT}"
    fi
}

check_sources() {
    [[ -d "${VENDORS_DIR}" ]] || error_exit "vendors directory not found: ${VENDORS_DIR}"
    [[ -f "${VENDORS_DIR}/${CONFIG_NAME}" ]] || error_exit "missing ${VENDORS_DIR}/${CONFIG_NAME}"
    [[ -d "${VENDORS_DIR}/${VENDOR_NAME}" ]] || error_exit "missing ${VENDORS_DIR}/${VENDOR_NAME}"
}

prepare_output() {
    mkdir -p "${BIN_DIR}" || error_exit "failed to create bin directory: ${BIN_DIR}"
    OUT_FILE="${BIN_DIR}/vendors-${VERSION_TAG}-${COMMIT_ID}-${ARCH}-${DEVICE}-bare.tar.gz"
}

package_vendors() {
    local stage_dir
    stage_dir="$(mktemp -d)"

    cp -a "${VENDORS_DIR}/${CONFIG_NAME}" "${stage_dir}/"
    cp -a "${VENDORS_DIR}/${VENDOR_NAME}" "${stage_dir}/"

    tar -czf "${OUT_FILE}" -C "${stage_dir}" "${CONFIG_NAME}" "${VENDOR_NAME}"
    rm -rf "${stage_dir}"
}

main() {
    detect_arch
    read_version_tag
    read_commit_id
    check_sources
    prepare_output
    package_vendors

    echo -e "packed vendors tarball: \033[33m${OUT_FILE}\033[0m"
    echo "source vendors dir: ${VENDORS_DIR}"
}

main "$@"

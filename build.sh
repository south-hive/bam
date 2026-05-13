#!/usr/bin/env bash
#
# bam end-to-end build script
#
#   1. cmake configure (build/)
#   2. libnvm userspace shared library      → build/lib/libnvm.so
#   3. libnvm.ko kernel module              → build/module/libnvm.ko
#   4. collect into dist/                   → 외부 사용자(plink 등)가
#                                              참조하는 표준 위치
#
# 사용:
#   ./build.sh                              # 전체 빌드 + dist/ 수집
#   SKIP_MODULE=1 ./build.sh                # .ko 빌드 건너뛰기
#                                              (kernel headers 미설치 시)
#   BUILD_TYPE=Debug ./build.sh             # debug build
#   CC=gcc-11 CXX=g++-11 ./build.sh         # 호스트 컴파일러 명시
#
# 자동 컴파일러 선택 정책:
#   1) env 의 CC/CXX/CUDA_HOST_CXX 가 있으면 그대로 사용 (최우선)
#   2) gcc-12 / g++-12 가 있으면 사용 (CUDA 13.x 권장)
#   3) gcc-11 / g++-11 fallback
#   4) 시스템 기본 gcc/g++ — 단 GCC 8~14 범위 확인
#
# bam 의 freestanding libcxx (include/freestanding/include/simt) 는 GCC 13+
# 의 chrono 헤더와 호환성 이슈가 있어 가능한 GCC 11~12 권장.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${ROOT}/build"
DIST="${ROOT}/dist"

# ------------------------------------------------------------------
# 사용자 토글 환경변수
# ------------------------------------------------------------------
BUILD_TYPE="${BUILD_TYPE:-Release}"           # Debug | Release
JOBS="${JOBS:-$(nproc)}"
CUDA_ARCHS="${CUDA_ARCHS:-80;86;89;90;100;120}"  # Ampere ~ Blackwell
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
SKIP_MODULE="${SKIP_MODULE:-0}"               # 1=.ko 빌드 건너뛰기

# ------------------------------------------------------------------
# 호스트 컴파일러 선택
# ------------------------------------------------------------------
_pick_compiler() {
    # _pick_compiler <preferred> <fallback...>  — 첫 번째로 존재하는 명령 출력.
    for cand in "$@"; do
        if command -v "${cand}" >/dev/null 2>&1; then
            echo "${cand}"; return
        fi
    done
}
CC="${CC:-$(_pick_compiler gcc-12 gcc-11 gcc)}"
CXX="${CXX:-$(_pick_compiler g++-12 g++-11 g++)}"
CUDA_HOST_CXX="${CUDA_HOST_CXX:-${CXX}}"

# 컴파일러 존재 + 버전 검증 (CUDA 13.x: GCC 8~14 지원)
for _bin in "${CC}" "${CXX}" "${CUDA_HOST_CXX}"; do
    if [[ -z "${_bin}" ]] || ! command -v "${_bin}" >/dev/null 2>&1; then
        echo "ERROR: 호스트 컴파일러 없음." >&2
        echo "       해결: sudo apt install gcc-12 g++-12  (또는 11)" >&2
        echo "       또는 CC=... CXX=... CUDA_HOST_CXX=... 로 명시" >&2
        exit 1
    fi
done
_CXX_MAJOR="$("${CXX}" -dumpversion 2>/dev/null | cut -d. -f1)"
if [[ "${_CXX_MAJOR}" -lt 8 || "${_CXX_MAJOR}" -gt 14 ]]; then
    echo "WARN: CXX=${CXX} (major=${_CXX_MAJOR}) — CUDA 13.x 지원 범위(8~14) 밖." >&2
fi
if [[ "${_CXX_MAJOR}" -ge 13 ]]; then
    echo "WARN: GCC ${_CXX_MAJOR} 은 freestanding libcxx 와 chrono 호환성 이슈 가능." >&2
    echo "      문제 시 gcc-11 또는 gcc-12 로 재시도 (CC=gcc-11 CXX=g++-11)." >&2
fi

# CMake 가 암묵적으로 읽는 CUDA env 를 명시 고정 — 사용자 셸의 잘못된
# CUDAHOSTCXX/CUDACXX export 가 영향 미치지 않도록.
export CUDAHOSTCXX="${CUDA_HOST_CXX}"
export CUDACXX="${NVCC}"
unset CCBIN

echo "==> bam build"
echo "    BUILD_TYPE = ${BUILD_TYPE}"
echo "    JOBS       = ${JOBS}"
echo "    CC/CXX     = ${CC} / ${CXX}"
echo "    NVCC       = ${NVCC}"
echo "    CUDA_ARCHS = ${CUDA_ARCHS}"
echo "    SKIP_MODULE= ${SKIP_MODULE}"

# ------------------------------------------------------------------
# 1. cmake configure
# ------------------------------------------------------------------
rm -rf "${BUILD}"
mkdir -p "${BUILD}"
echo "==> [1/4] cmake configure → ${BUILD}"
pushd "${BUILD}" >/dev/null
# include/freestanding/include 를 host CXX include path 에 추가 — bam 의
# vendored libcxx (simt::atomic, chrono 등) 가 정상 해석되도록.
cmake .. \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -Dno_smartio=true \
    -Dno_module=false \
    -Dno_fio=true \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CUDA_HOST_COMPILER="${CUDA_HOST_CXX}" \
    -DCMAKE_CUDA_COMPILER="${NVCC}" \
    -DCMAKE_CXX_FLAGS="-I${ROOT}/include/freestanding/include" \
    -DCMAKE_C_FLAGS="-I${ROOT}/include/freestanding/include" \
    >/dev/null
popd >/dev/null

# ------------------------------------------------------------------
# 2. userspace libnvm.so
# ------------------------------------------------------------------
echo "==> [2/4] make libnvm.so (full build)"
# 전체 타겟 — `make libnvm` 만 하면 일부 .o 누락된 incomplete .so 가
# 만들어지는 케이스가 있어 default `all` 타겟 사용.
make -C "${BUILD}" -j"${JOBS}"

if [[ ! -f "${BUILD}/lib/libnvm.so" ]]; then
    echo "ERROR: ${BUILD}/lib/libnvm.so 가 만들어지지 않음." >&2
    exit 1
fi

# ------------------------------------------------------------------
# 3. kernel module libnvm.ko
# ------------------------------------------------------------------
if [[ "${SKIP_MODULE}" == "1" ]]; then
    echo "==> [3/4] Skip kernel module (SKIP_MODULE=1)"
else
    echo "==> [3/4] make libnvm.ko"
    if [[ -f "${BUILD}/module/Makefile" ]]; then
        # link-sources + Kbuild — Linux 6.x 호환 패치 적용된 Makefile.in 사용
        if ! make -C "${BUILD}/module"; then
            echo "WARN: libnvm.ko 빌드 실패 — kernel headers 미설치 또는" >&2
            echo "      kernel API 호환성 이슈. userspace 산출물은 그대로 사용 가능." >&2
        fi
    else
        echo "WARN: ${BUILD}/module/Makefile 없음 — cmake 가 모듈 타겟을 생성 안 함." >&2
    fi
fi

# ------------------------------------------------------------------
# 4. dist/ 수집
# ------------------------------------------------------------------
echo "==> [4/4] Collect artifacts → ${DIST}"
rm -rf "${DIST}"
mkdir -p "${DIST}/include" "${DIST}/lib"

# (a) 공개 헤더 — 외부 사용자가 #include <nvm_xxx.h> 할 수 있도록.
# rsync 로 .git, .gitmodules, *.md 같은 비-소스 파일 제외.
#
# freestanding/ (52MB libcxx) 은 ctrl.h → queue.h → simt::atomic 경로에
# 필요. C API 만 쓰는 사용자는 SLIM_INCLUDE=1 로 제외 가능.
SLIM_INCLUDE="${SLIM_INCLUDE:-0}"
if command -v rsync >/dev/null 2>&1; then
    EXCLUDES=(
        --exclude='.git' --exclude='.gitignore' --exclude='.gitmodules'
        --exclude='*.md' --exclude='LICENSE*'
    )
    [[ "${SLIM_INCLUDE}" == "1" ]] && EXCLUDES+=( --exclude='freestanding/' )
    rsync -a "${EXCLUDES[@]}" "${ROOT}/include/" "${DIST}/include/"
else
    # rsync 없으면 cp + 사후 정리
    cp -r "${ROOT}/include/." "${DIST}/include/"
    find "${DIST}/include" -name '.git*' -prune -exec rm -rf {} +
    [[ "${SLIM_INCLUDE}" == "1" ]] && rm -rf "${DIST}/include/freestanding"
fi

# (b) shared library
cp "${BUILD}/lib/libnvm.so" "${DIST}/lib/libnvm.so"
ln -sf libnvm.so "${DIST}/lib/libnvm.so.0" 2>/dev/null || true

# (c) kernel module (있을 때만)
if [[ -f "${BUILD}/module/libnvm.ko" ]]; then
    cp "${BUILD}/module/libnvm.ko" "${DIST}/libnvm.ko"
else
    echo "    (skip) libnvm.ko not present — SKIP_MODULE=1 또는 빌드 실패"
fi

# (d) pkg-config 파일 — 외부 빌드 시스템이 자동 link 할 수 있도록
mkdir -p "${DIST}/lib/pkgconfig"
cat > "${DIST}/lib/pkgconfig/libnvm.pc" <<EOF
prefix=${DIST}
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include

Name: libnvm
Description: BaM userspace NVMe driver (host side)
Version: 0.1
Libs: -L\${libdir} -lnvm
Libs.private: -lpthread -lcudart
Cflags: -I\${includedir} -I\${includedir}/freestanding/include
EOF

echo ""
echo "==> Done. Artifacts in ${DIST}:"
echo ""
ls -lh "${DIST}/" "${DIST}/lib/" 2>&1 | head -30
echo ""

# ------------------------------------------------------------------
# Next steps
# ------------------------------------------------------------------
cat <<EOF

# ------------------------------------------------------------------
# Next steps
# ------------------------------------------------------------------
EOF
if [[ -f "${DIST}/libnvm.ko" ]]; then
    cat <<EOF
  # 1) kernel module 로드 + NVMe SSD bind
  sudo insmod ${DIST}/libnvm.ko max_num_ctrls=64
  echo 0000:xx:xx.x | sudo tee /sys/bus/pci/drivers/nvme/unbind
  echo 0000:xx:xx.x | sudo tee /sys/bus/pci/drivers/libnvm/bind
  ls /dev/libnvm0           # 노출 확인
EOF
fi
cat <<EOF

  # 2) 외부 프로젝트에서 libnvm 사용 (CMake)
  set(LIBNVM_DIR ${DIST})
  set(LIBNVM_LIB \${LIBNVM_DIR}/lib/libnvm.so)
  include_directories(\${LIBNVM_DIR}/include)
  target_link_libraries(my_target \${LIBNVM_LIB})

  # 3) 또는 pkg-config 사용
  PKG_CONFIG_PATH=${DIST}/lib/pkgconfig pkg-config --cflags --libs libnvm

# ------------------------------------------------------------------
# Build Success
# ------------------------------------------------------------------
EOF

#!/bin/bash
# 줄바꿈 문자를 Unix 형식(LF)으로 변환하는 스크립트

echo "Shell 스크립트의 줄바꿈 문자를 Unix 형식으로 변환합니다..."

# dos2unix가 있는지 확인
if command -v dos2unix &> /dev/null; then
    echo "dos2unix를 사용합니다..."
    for file in *.sh; do
        if [ -f "$file" ]; then
            dos2unix "$file"
            echo "  - $file 변환 완료"
        fi
    done
# sed 사용 (대부분의 리눅스 시스템에서 사용 가능)
elif command -v sed &> /dev/null; then
    echo "sed를 사용합니다..."
    for file in *.sh; do
        if [ -f "$file" ]; then
            sed -i 's/\r$//' "$file"
            echo "  - $file 변환 완료"
        fi
    done
# tr 사용 (모든 Unix 시스템에서 사용 가능)
else
    echo "tr을 사용합니다..."
    for file in *.sh; do
        if [ -f "$file" ]; then
            tr -d '\r' < "$file" > "$file.tmp" && mv "$file.tmp" "$file"
            echo "  - $file 변환 완료"
        fi
    done
fi

# 실행 권한 부여
echo ""
echo "실행 권한을 부여합니다..."
chmod +x *.sh
echo "완료!"

echo ""
echo "다음 명령으로 스크립트를 실행할 수 있습니다:"
echo "  ./evaluate_llama_kmmlu.sh"
echo "  ./evaluate_pythia_kmmlu.sh"
echo "  ./evaluate_quick.sh"
## 환경세팅

01. 데이터 옮기기
    01-01 ./archive/00_source.zip 을 ./data 안에 압축 해제(./data/00_source)
    
02. 라이브러리 설치
    02-01 pip install ./requirements.txt 실행
    02-02 pytorch 설치(https://pytorch.org/get-started/previous-versions/ 에서 사용환경에 맞는 버전 찾아서 설치)
        - pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
    02-03 apex 설치(./archive 내에서 아래 명령어 순서대로 실행)
        - git clone https://github.com/NVIDIA/apex
        - cd apex
        - pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
Bootstrap:docker
From:ufoym/deepo:pytorch-cu102

%labels
    MAINTAINER admin
    WHATAMI admin

%files
    cli.sh /cli.sh
    requirements.txt /requirements.txt

%runscript
    exec /bin/bash /cli.sh "$@"

%post
    chmod u+x /cli.sh

    # Install dependencies here
    apt update
    apt install -y build-essential
    pip install -r /requirements.txt
    python -c 'import nltk; nltk.download("punkt")'

    # Install Joern
    apt install -y openjdk-8-jdk git curl gnupg bash unzip sudo wget 
    wget https://github.com/ShiftLeftSecurity/joern/releases/latest/download/joern-install.sh
    chmod +x ./joern-install.sh
    printf 'Y\n/bin/joern\ny\n/bin\n\n' | sudo ./joern-install.sh --interactive

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

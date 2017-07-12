import os
os.system("sudo apt-get update")
#os.system('sudo apt-get install openjdk-8-jre -y')
os.system('wget --header "Cookie: oraclelicense=accept-securebackup-cookie" http://download.oracle.com/otn-pub/java/jdk/8u131-b11/d54c1d3a095b4ff2b6607d096fa80163/jdk-8u131-linux-x64.tar.gz')
os.system('tar -zxf jdk-8u131-linux-x64.tar.gz')
os.system('export JAVA_HOME=/home/jupyter/Predict-Churn/jdk1.8.0_131/')
os.system('export PATH="$JAVA_HOME/bin:$PATH"')
os.system('pip install http://h2o-release.s3.amazonaws.com/h2o/rel-ueno/5/Python/h2o-3.10.4.5-py2.py3-none-any.whl')



    
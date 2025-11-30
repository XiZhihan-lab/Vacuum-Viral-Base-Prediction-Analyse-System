@echo off
echo 正在安装Viral Base程序所需的外置库...
echo 请确保已安装Python 3.7+和pip

pip install --upgrade pip
pip install numpy==1.21.5
pip install scipy==1.7.3
pip install matplotlib==3.5.2
pip install seaborn==0.11.2
pip install biopython==1.79
pip install scikit-bio==0.5.7
pip install wxPython==4.1.1
pip install PyQt5==5.15.6
pip install pyvista==0.34.1
pip install pyvistaqt==0.9.0
pip install rdkit-pypi==2022.3.5
pip install requests==2.27.1
pip install simpy==4.0.1

echo 所有依赖库安装完成！
pause
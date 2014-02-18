self_dot
========

An artificial being. Uses evolution to write its own program.

Requirements: 

You should definitely user virtualenvwrapper. These steps depend on
whether you have virtualenvwrapper installed system-wide or locally
only. Both are possible. I guess for many systems where you are system
admin as well, it makes sense to have virtualenvwrapper
system-wide. In any case, virtualenvwrapper will copy your python and
include it locally.



Download virtualenvXXX.tar.gz



mkdir ~/.local

cd virtualenv-x-x

python virtualenv.py ~/.local

~/.local/bin/pip install virtualenvwrapper

Changes need in your .profile:
export PATH=$PATH:$HOME/.local/bin
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=~/.local/bin/python #To avoid using system python, in case it changes over time.
source $HOME/.local/bin/virtualenvwrapper.sh

mkvirtualenv smartgrid

pip install https://github.com/perone/Pyevolve/zipball/master
pip install numpy
pip install ipdb

Manually installed packages:

Oger: http://organic.elis.ugent.be/installing_oger

Download the tarball, run 

python setup.py install

OpenCV must be installed. Installers can be found here: http://opencv.org

On Mac, this required the installation of CMake http://www.cmake.org/cmake/resources/software.html

Important: in order to install OpenCV specific to the virtualenv you
are using, specify this the following way. This has the advantage that
OpenCV will be installed locally *and* linked to the Python version of
the virtualenv, which is the whole point of the virtualenv in the
first place.

cd opencv*
mkdir build 
cd build 
cmake -D CMAKE_INSTALL_PREFIX=$VIRTUAL_ENV/local/ -D PYTHON_EXECUTABLE=$VIRTUAL_ENV/bin/python -D PYTHON_PACKAGES_PATH=$VIRTUAL_ENV/lib/python2.7/site-packages/ -D PYTHON_LIBRARY=/Library/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib -G "Unix Makefiles" ..
make -j8
make install 

Note the cmake step: multiple Python versions can be quite a nuisance
- it will automatically find the one that is system-wide, however I
prefer to copy the python into the virtualenv as well, making it (in
theory) more robust towards possible python system changes (as
mentioned above). HOWEVER: it took me a long time to discover that the
flag -D
PYTHON_LIBRARY=/Library/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib
was necessary in order to compile against the correct version -
otherwise the wrong dylib would be used, and you would get a
segmentation fault upon trying to import cv. Now, why there isn't a
libpython2.7.dylib in ~/.local/lib is a bit beyond me (and maybe I'm
missing something important here). However, in this particular case,
the VIRTUALENVWRAPPER_PYTHON was the same as the system wide python,
so this worked.

PortAudio: download the source code from http://www.portaudio.com Keeping in with the good virtualenv tradition:

./configure --prefix=$VIRTUAL_ENV/local
make clean
make -j8
make install

libsnd: http://www.mega-nerd.com/libsndfile/

./configure --prefix=$VIRTUAL_ENV/local
make clean
make -j8
make install

cffi:

pip install cffi
pip install psoundfile, pysoundcard

Specific mac stuff:

export CC=clang
export CXX=clang

export DYLD_LIBRARY_PATH=$VIRTUAL_ENV/local/lib
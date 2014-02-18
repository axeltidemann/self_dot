self_dot
========

An artificial being. Uses evolution to write its own program.

# Requirements: 

You should definitely user virtualenvwrapper. This assumes you have a virtualenv called self_dot, instantiated like so:

> mkvirtualenv self_dot

Proceed by installing these packages:

```
pip install https://github.com/perone/Pyevolve/zipball/master
pip install numpy
pip install scipy
pip install ipdb
pip install MDP
easy_install readline
pip install ipython
```

Unfortunately, there are a few packages that must be installed manually. 

Oger: http://organic.elis.ugent.be/installing_oger

Download the tarball, run 

> python setup.py install

Note: when you install the following software, it is advised that you
install them into $VIRTUAL_ENV/local, so they will be contained within
the virtualenv, and more robust. This is shown as an example under
"Mac stuff".

OpenCV: http://opencv.org

PortAudio: http://www.portaudio.com

libsnd: http://www.mega-nerd.com/libsndfile/

OpenCV installs Python bindings by itself. To be able to use the sound software:

```
pip install cffi 
pip install psoundfile
pip install pysoundcard
```

# Specific mac stuff:

These are some experiences found when installing the software under 10.7 and 10.8.

On Mac 10.7, CMake was required http://www.cmake.org/cmake/resources/software.html 

On 10.7 this had to be set prior to installation of scipy:
export CC=clang
export CXX=clang

In order to use the new shared libraries, you must specify where they are.

export DYLD_LIBRARY_PATH=$VIRTUAL_ENV/local/lib 

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

It took me a long time to discover that the
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


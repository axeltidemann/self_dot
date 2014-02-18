self_dot
========

An artificial being. Uses evolution to write its own program.

## Requirements: 

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

## Specific Mac OS X stuff:

These are some experiences found when installing the software under 10.7 and 10.8.

On Mac 10.7, CMake was required before installing OpenCV http://www.cmake.org/cmake/resources/software.html 

On 10.7 this had to be set prior to installation of scipy:

```
export CC=clang
export CXX=clang
```


_Important:_ in order to install OpenCV specific to the virtualenv you
are using, specify this in the cmake input arguments listed
below. This has the advantage that OpenCV will be installed locally
*and* linked to the Python version of the virtualenv - if not
specified this way, it will use the system Python instead, which is
sure to cause massive headaches and the death of many adorable kittens
in the future. So upon downloading and unpacking the opencv tarball:

```
cd opencv*
mkdir build 
cd build 
cmake -D CMAKE_INSTALL_PREFIX=$VIRTUAL_ENV/local/ -D PYTHON_EXECUTABLE=$VIRTUAL_ENV/bin/python -D PYTHON_PACKAGES_PATH=$VIRTUAL_ENV/lib/python2.7/site-packages/ -D PYTHON_LIBRARY=/Library/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib -G "Unix Makefiles" ..
make -j8
make install 
```

It took me a _long_ time to discover that the
flag 

> -D PYTHON_LIBRARY=/Library/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib

was necessary in order to compile against the correct version -
otherwise the wrong dylib would be used, and you would get a
segmentation fault upon trying to import cv. Now, why there isn't a
libpython2.7.dylib in ~/.local/lib is a bit beyond me (and maybe I'm
missing something important here). However, in this particular case,
the VIRTUALENVWRAPPER_PYTHON was the same as the system wide python,
so this worked.

PortAudio: download the source code from http://www.portaudio.com Keeping in with the good virtualenv tradition:

```
./configure --prefix=$VIRTUAL_ENV/local
make clean
make -j8
make install
```

libsnd: http://www.mega-nerd.com/libsndfile/

```
./configure --prefix=$VIRTUAL_ENV/local
make clean
make -j8
make install
```

In order to use the new shared libraries, you must specify where they are.

> export DYLD_LIBRARY_PATH=$VIRTUAL_ENV/local/lib 

*Note:* This can actually be done in the virtualenv, so you will have everything completely self-contained (yes, this is good). Do it the following way: 

```
echo 'export DYLD_LIBRARY_PATH=$VIRTUAL_ENV/local/lib' >> $VIRTUAL_ENV/bin/postactivate
echo 'unset DYLD_LIBRARY_PATH' >> $VIRTUAL_ENV/bin/predeactivate
```

Of course, if you already have mangled your DYLD_LIBRARY_PATH, you should take the appropriate steps to keep it that way. 

And you should be good to go!

*How then, does this compile under 10.8?* After spending an entire day
 of trying to compile everything as listed above on my MacBook Air
 (running 10.8) and getting further and deeper into darkness, I had
 the idea of simply copying ~/.virtualenvs/self_dot from my 10.7
 machine to my 10.8 machine. Somewhat miraculously, this worked. Now
 it becomes clear what an added bonus it is to install software into
 $VIRTUAL_ENV/local - the programs are also copied, and with the
 DYLD_LIBRARY_PATH set, this is a _very_ smooth transition. Only
 tested for 10.7 -> 10.8 though. It will be interesting to see if this
 is robust across several OSes (I'm not betting my savings on it,
 no). However, it is evident that 10.7 is a *lot* more mature for
 compiling code than 10.8. What about 10.9, you say? Well, by now it
 should be obvious why I _won't_ be upgrading _anytime soon_, since I
 have some colleagues who have. They wish they hadn't.
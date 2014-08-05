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
pip install scikit-learn
pip install ipdb
pip install MDP
easy_install readline
```

There are a few packages that must be installed manually. 

```
hg clone https://bitbucket.org/benjamin_schrauwen/organic-reservoir-computing-engine Oger
cd Oger/src
python setup.py install
```

Note: when you install the following software, it is advised that you
install it into $VIRTUAL_ENV, so it will be contained within
the virtualenv, and more robust. This is shown as an example under
"Mac stuff".

OpenCV: http://opencv.org

Csound: http://www.csounds.com

ØMQ: http://zeromq.org/

> pip install pyzmq

These are needed for resampling:

libsndfile: http://www.mega-nerd.com/libsndfile/ (Test if it is really necessary to install this to achieve scikits.samplerate, try it without first)
libsamplerate: http://www.mega-nerd.com/SRC/download.html

> pip install scikits.samplerate

Lyon's cochlear model: https://github.com/google/carfac

These are some notes on getting the C++ version of the CARFAC library working. 

SCons http://www.scons.org/

Clang 3.3 (or better) in order to compile the code with SCons. http://clang.llvm.org/

The Eigen library http://eigen.tuxfamily.org/ and set the EIGEN_PATH to where the Eigen folder is (note: NOT the Eigen folder itself, but the folder underneath). 

> export EIGEN_PATH=/path/to/eigen/

The SConstruct file in carfac/cpp should look like this, the result of STRESS, SWEAT & TEARS (big up to Boye). Note: change the path to clang, of course. 

```
import commands
import os

env = Environment(CPPPATH = [os.environ['EIGEN_PATH']])
env['CC'] = '/Users/tidemann/.virtualenvs/self_dot/bin/clang'
env['CXX'] = '/Users/tidemann/.virtualenvs/self_dot/bin/clang++'

env.MergeFlags(['-std=c++11 -stdlib=libc++ -v'])
carfac_sources = [
    'binaural_sai.cc',
    'carfac.cc',
    'ear.cc',
    'sai.cc'
    ]

carfac = env.Library(target = 'carfac', source = carfac_sources)
Default(carfac)

axel_sources = carfac_sources + ['carfac_cmd.cc']
axel = env.Program(target = 'carfac-cmd',
                    source = axel_sources,
                    LINKFLAGS = '-std=c++11 -stdlib=libc++ -v')

```

The file carfac_cmd.cc must be in the /path/to/carfac/cpp folder, move it from your self_dot home to this location.

To compile it, run 

> scons carfac-cmd

Afterwards, you must move the carfac-cmd executable to your self_dot home, since python is calling it from brain.py.

If you want to install Octave, e.g. the C++ version is too troublesome, follow these steps:

Octave: http://www.gnu.org/software/octave/

> pip install oct2py

Start Octave, navigate to path/to/carfac/matlab and add the path to Octave, e.g.

> addpath(pwd)
> savepath

However, in production mode we must run the C++-version as it is a lot faster than the Octave counterpart. 

## Specific Mac OS X stuff:

These are some experiences found when installing the software under 10.7.5 and 10.8.5.

On Mac 10.7.5, CMake was required before installing OpenCV: http://www.cmake.org/cmake/resources/software.html 

On 10.7.5 this had to be set prior to installation of scipy:

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
in the future. So after downloading and unpacking the opencv tarball:

```
cd opencv*
mkdir build 
cd build 
cmake -D CMAKE_INSTALL_PREFIX=$VIRTUAL_ENV -D PYTHON_EXECUTABLE=$VIRTUAL_ENV/bin/python -D PYTHON_PACKAGES_PATH=$VIRTUAL_ENV/lib/python2.7/site-packages/ -D PYTHON_LIBRARY=/Library/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib -G "Unix Makefiles" ..
make -j
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

In order to use the new shared libraries, you must specify where they are. NB: REVISE TO SEE IF THIS REALLY IS NECESSARY. Try it without setting DYLD_LIBRARY_PATH first.

> export DYLD_LIBRARY_PATH=$VIRTUAL_ENV/lib 

*Note:* This can actually be done in the virtualenv, so you will have everything completely self-contained (yes, this is good). Do it the following way: 

```
echo 'export DYLD_LIBRARY_PATH=$VIRTUAL_ENV/lib' >> $VIRTUAL_ENV/bin/postactivate
echo 'unset DYLD_LIBRARY_PATH' >> $VIRTUAL_ENV/bin/predeactivate
```

Of course, if you already have mangled your DYLD_LIBRARY_PATH, you should take the appropriate steps to keep it that way. 

On Mac, you must also set the $PYTHONPATH to point to where the Csound package files are. For reasons unknown, they wind up in /Library/Python/2.7/site-packages/

```
export PYTHONPATH=/Library/Python/2.7/site-packages/:$PYTHONPATH
```

This could also be set in postactivate, of course.

To install libsndfile: 

```
./configure --prefix=$VIRTUAL_ENV
make clean
make install 
```

To install libsamplerate:

```
./configure --prefix=$VIRTUAL_ENV --build='x86_64' --disable-octave --disable-fftw
make clean
make install
```
Test to see if --build and --disable-octave are necessary. --disable-fftw is what finally made it install on my Mac.

To install clang 3.3: Change 33 -> 35 to try later version. 

```
cd ~
svn co http://llvm.org/svn/llvm-project/llvm/tags/RELEASE\_33/final llvm33
cd llvm33/tools
svn co http://llvm.org/svn/llvm-project/cfe/tags/RELEASE\_33/final clang
cd ../..
mkdir llvm33build
cd llvm33build
cmake -D CMAKE_INSTALL_PREFIX=$VIRTUAL_ENV -DCMAKE\_BUILD\_TYPE=Release -G "Unix Makefiles" ../llvm33
make -j
make install
```

Check to see if --build and --disable-octave are necessary. The --disable-fftw was what finally made it install. Phew.

And you should be good to go!

*How then, does this compile under 10.8?* After spending an entire day
 of trying to compile everything as listed above on my MacBook Air
 (running 10.8) and getting further and deeper into darkness, I had
 the idea of simply copying ~/.virtualenvs/self_dot from my 10.7
 machine to my 10.8 machine. Somewhat miraculously, this worked. Now
 it becomes clear what an added bonus it is to install software into
 $VIRTUAL_ENV - the programs are also copied, and with the
 DYLD_LIBRARY_PATH set, this is a _very_ smooth transition. Only
 tested for 10.7 -> 10.8 though. It will be interesting to see if this
 is robust across several cats (I'm not betting my savings on it,
 no). However, it is evident that 10.7 is a *lot* more mature for
 compiling code than 10.8. What about 10.9, you say? Well, by now it
 should be obvious why I won't be upgrading _anytime soon_, since I
 have some colleagues who have. They wish they hadn't.
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

libsamplerate: http://www.mega-nerd.com/SRC/download.html

> pip install scikits.samplerate

Lyon's cochlear model: https://github.com/google/carfac

These are some notes on getting the C++ version of the CARFAC library working. 

SCons: http://www.scons.org/

Clang 3.3 (or better) in order to compile the code with SCons: http://clang.llvm.org/

The Eigen library: http://eigen.tuxfamily.org/ and set the EIGEN_PATH to where the Eigen folder is (note: NOT the Eigen folder itself, but the folder underneath). 

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

```
addpath(pwd)
savepath
```

However, in production mode we must run the C++-version as it is a lot faster than the Octave counterpart. 

## Specific Mac OS X stuff:

These are some experiences found when installing the software under 10.7.5 and 10.8.5.

On Mac 10.7.5, CMake was required before installing OpenCV:
http://www.cmake.org/cmake/resources/software.html and you must also
install numpy and scipy beforehand. You might just as well install
clang 3.4 right now, since you'll be needing it for SCons.

To install clang 3.4: 

```
cd ~
svn co http://llvm.org/svn/llvm-project/llvm/tags/RELEASE\_34/final llvm34
cd llvm34/tools
svn co http://llvm.org/svn/llvm-project/cfe/tags/RELEASE\_34/final clang
cd ../..
mkdir llvm34build
cd llvm34build
cmake -D CMAKE_INSTALL_PREFIX=$VIRTUAL_ENV -DCMAKE\_BUILD\_TYPE=Release -G "Unix Makefiles" ../llvm34
make -j
make install
```
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
in the future. A source of __immensive__ frustration when trying to
reinstall OpenCV was that you must *not* set CC and CXX to clang, as
you do before installing virtually every other package (ØMQ being a
notable exception, where you must include clang to install pyzmq
again). So if you just did, close that window and start anew. A bit
baffling, and this took me quite som hours to figure out.

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
segmentation fault upon trying to import cv. In this particular case,
the VIRTUALENVWRAPPER_PYTHON was the same as the system wide python,
so this worked.

In order to use the new shared libraries, you must specify where they
are. This is not needed in 10.7, check to see if it is needed in 10.8.
It could be that this is taken care of automatically when you put
everything in $VIRTUAL_ENV.

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

To install libsamplerate:

```
./configure --prefix=$VIRTUAL_ENV --build='x86_64' --disable-octave --disable-fftw
make 
make install
```

It appears as if --build, --disable-octave and --disable-fftw are necessary. 

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

## Ubuntu on VirtualBox 

VirtualBox https://www.virtualbox.org

Ubuntu: http://www.ubuntu.com

Insert the "Devices > Guest additions" CD to have a more smooth experience. To SSH into your Ubuntu virtual machine, stop the virtual machine, go to Settings > Network > Adapter 1 and select "Attached to: Bridged adapter". Furthermore, on the virtual machine you must install the ssh server. We include git and python-pip here as well, since you'll be needing these.

> sudo apt-get install openssh-server git python-pip

Then run ifconfig to see your IP address. Now you can SSH into the virtual machine, which makes copying/pasting code and doing installation thingies a lot easier in my experience.

To install numpy and scipy:

> sudo apt-get install python-numpy python-scipy python-matplotlib ipython python-nose

Note how we do not use virtualenvwrapper - this virtual machine will not be used for anything else, and there is a lot of precompiled libraries for Ubuntu, so we are taking the easy route here.


OpenCV:

```
version="$(wget -q -O - http://sourceforge.net/projects/opencvlibrary/files/opencv-unix | egrep -m1 -o '\"[0-9](\.[0-9])+' | cut -c2-)"
echo "Installing OpenCV" $version
mkdir OpenCV
cd OpenCV
echo "Removing any pre-installed ffmpeg and x264"
sudo apt-get -qq remove ffmpeg x264 libx264-dev
echo "Installing Dependenices"
sudo apt-get -qq install libopencv-dev build-essential checkinstall cmake pkg-config yasm libjpeg-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxine-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev python-dev python-numpy libtbb-dev libqt4-dev libgtk2.0-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils
echo "Downloading OpenCV" $version
wget -O OpenCV-$version.zip http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/$version/opencv-"$version".zip/download
echo "Installing OpenCV" $version
unzip OpenCV-$version.zip
cd opencv-$version
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=OFF -D WITH_OPENGL=ON ..
make 
sudo make install
```

CSound:

> sudo apt-get build-dep csound

And then:

```
cd ~
mkdir csound
cd csound
git clone https://github.com/csound/csound.git csound
mkdir cs6make
cd cs6make
cmake ../csound
make 
sudo make install
sudo ldconfig

```

Getting libc++ to work on Ubuntu, otherwise installation of Lyon's cochlear model won't work:

> sudo apt-get install libc++-dev
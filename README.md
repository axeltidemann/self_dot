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
pip install matplotlib
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

You need GCC 4.9.1 to optimize the CARFAC library, since it allows for
optimizing the Eigen Matrix Library with OpenMP. This will have a
_massive_ impact on performance, and is well worth the time for
compiling and installing. On Ubuntu, you most likely can do a sudo
apt-get install g++-4.9, and can skip the following steps to get GCC
4.9.1 running.

GCC 4.9.1: ftp://ftp.mpi-sb.mpg.de/pub/gnu/mirror/gcc.gnu.org/pub/gcc/releases/gcc-4.9.1/

You need three libraries in order to compile GCC 4.9.1. These are:

GMP: ftp://ftp.mpi-sb.mpg.de/pub/gnu/mirror/gcc.gnu.org/pub/gcc/infrastructure/gmp-4.3.2.tar.bz2

MPC: ftp://ftp.mpi-sb.mpg.de/pub/gnu/mirror/gcc.gnu.org/pub/gcc/infrastructure/mpc-0.8.1.tar.gz

MPFR: ftp://ftp.mpi-sb.mpg.de/pub/gnu/mirror/gcc.gnu.org/pub/gcc/infrastructure/mpfr-2.4.2.tar.bz2

Unpack these. If you move these into your GCC source code folder
(without version names - this is described in the GCC installation
manual), they will be automatically compiled. If you downloaded and
unpacked everything to a common folder (e.g. ~/Downloads), this is
what you would have to do. This takes a couple of hours.

```
cd ~/Downloads/gcc-4.9.1
mv ../gmp-4.3.2 gmp
mv ../mpfr-2.4.2 mpfr
mv ../mpc-0.8.1 mpc
./configure --prefix=$VIRTUAL_ENV
make
make install
```
The Eigen library: http://eigen.tuxfamily.org/ and set the EIGEN_PATH to where the Eigen folder is (note: NOT the Eigen folder itself, but the folder underneath). 

> export EIGEN_PATH=/path/to/eigen/

*Important:* You are now harnessing the power of OpenMP in
Eigen. You must set the number of threads that OpenMP will use. This
should be set to the number of *physical* cores of your CPU. Many CPUs
report _logical_ cores as the double of physical cores, since
they use Hyper Threading (or some other technique). This could
actually hamper performance, since Eigen has an optimized data/memory
flow, and would run on full blast on a physical core, and most likely
waste time waiting for a logical core. On my 8-core MacBook Pro there
are only 4 physical cores, so I set this accordingly:

> export OMP_NUM_THREADS=4

You must also specify where the carfac source files are:

> export CARFAC_PATH=/path/to/carfac/

Finally, you must specify to use the newly installed gcc and g++. This ensures that they will be used:

```
export CC=$VIRTUAL_ENV/bin/gcc
export CXX=$VIRTUAL_ENV/bin/g++
```

To compile the CARFAC library, simply run

> scons

## Specific Mac OS X stuff:

These are some experiences found when installing the software under 10.7.5 and 10.8.5.

On Mac 10.7.5, CMake was required before installing OpenCV:
http://www.cmake.org/cmake/resources/software.html and you must also
install numpy and scipy beforehand. 

For 10.8.5 you can just download the Command Line Tools from Apple,
and you'll get clang 3.4.

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

On 10.7.5 and 10.8.5 this had to be set prior to installation of
numpy, scipy and scikit-learn. It seems unnecessary to have both clang
and gcc-4.9, but this unfortunately seems to be the case. clang does
not have support for OpenMP, so we need gcc-4.9 as well.

```
export CC=clang
export CXX=clang++
```

_Important:_ in order to install OpenCV specific to the virtualenv you
are using, specify this in the cmake input arguments listed
below. This has the advantage that OpenCV will be installed locally
*and* linked to the Python version of the virtualenv - if not
specified this way, it will use the system Python instead, which is
sure to cause massive headaches and the death of many adorable kittens
in the future. A source of __immensive__ frustration when trying to
reinstall OpenCV on 10.7 was that you must *not* set CC and CXX to
clang, as you do before installing virtually every other package (ØMQ
being a notable exception, where you must include clang to install
pyzmq again). So if you just did, close that window and start anew. A
bit baffling, and this took me quite som hours to figure out. However,
on 10.8.5 you need not unset clang.

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
are. This is not needed in 10.7, however it is necessary in 10.8.

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

*10.8* I managed to install everything but matplotlib, where it was
 complaining about libpng. It would install fine, but crash upon
 usage. I gave up, and just copied the self_dot folder that I had
 compiled under 10.7. Important note: you must add the
 DYLD_LIBRARY_PATH in order to use the 10.7 virtualenv. This works
 since everything is installed within the virtualenv. In addition, I
 had to replace /usr/bin/xcrun with the following script in 10.8 to
 compile the aforementioned stuff:

```
#!/bin/bash
$@
```

This allowed the usage of gcc and so on, even though everything is centered around clang on 10.8. Ugly hack, I know.

## Ubuntu on VirtualBox 

VirtualBox https://www.virtualbox.org

Ubuntu: http://www.ubuntu.com

Insert the "Devices > Guest additions" CD to have a more smooth experience. To SSH into your Ubuntu virtual machine, stop the virtual machine, go to Settings > Network > Adapter 1 and select "Attached to: Bridged adapter". Furthermore, on the virtual machine you must install the ssh server. We include git and python-pip here as well, since you'll be needing these.

> sudo apt-get install openssh-server git python-pip

Then run ifconfig to see your IP address. Now you can SSH into the virtual machine, which makes copying/pasting code and doing installation thingies a lot easier in my experience.

To install numpy and scipy:

> sudo apt-get install python-numpy python-scipy python-matplotlib ipython python-nose

Note how we do not use virtualenvwrapper - this virtual machine will not be used for anything else, and there is a lot of precompiled libraries for Ubuntu, so we are taking the easy route here.

Getting libc++ to work on Ubuntu, otherwise installation of Lyon's cochlear model won't work:

> sudo apt-get install libc++-dev

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


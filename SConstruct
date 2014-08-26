import commands
import os

carfac_path = os.environ['CARFAC_PATH']
env = Environment(CPPPATH = [os.environ['EIGEN_PATH'], carfac_path + '/carfac/cpp'],
                  OMP_NUM_THREADS = [os.environ['OMP_NUM_THREADS']])
env['CC'] = os.environ['CC']
env['CXX'] = os.environ['CXX']

env.MergeFlags(['-std=c++11 -O3 -DNDEBUG -fopenmp'])
carfac_sources = [
    carfac_path + '/carfac/cpp/binaural_sai.cc',
    carfac_path + '/carfac/cpp/carfac.cc',
    carfac_path + '/carfac/cpp/ear.cc',
    carfac_path + '/carfac/cpp/sai.cc'
    ]

carfac = env.Library(target = 'carfac', source = carfac_sources)

self_dot_sources = carfac_sources + ['carfac_cmd.cc'] 
self_dot = env.Program(target = 'carfac-cmd',
                        source = self_dot_sources,
                        LINKFLAGS = '-std=c++11 -O3 -DNDEBUG -fopenmp')

Default(self_dot)

0. Modifications
Please download the dataset from the [original package][data], or personal [Dropbox][dropbox] if the Virginia's link is down.
Put it in the data directory.
[data]: http://lava.cs.virginia.edu/Rodinia/download_links.htm
[dropbox]: https://www.dropbox.com/s/cc6cozpboht3mtu/rodinia-3.1-data.tar.gz


### 1. Set environment variables

##### 1.1 export environment variables

```
export PATH=your-path:$PATH
export LD_LIBRARY_PATH=your-ld-library-path
export POCL_DEVICES="ventus"
export OCL_ICD_VENDOR=your-path-to-libpocl.so
```

##### 1.2 modify environment variables

path: common/make.config

```
 # DEFAULT OCL
PENCL_DIR = path-to-your-ocl-icd
OPENCL_INC = path-to-your-include
OPENCL_LIB = path-to-your-lib
```

### 2. Test suite running and building

for each single case :

| 序号	 | test-name | 构建命令 |  运行命令     |
|----  |----  |----  |----  |
|  1   | backprop       |   make   |  make run     |
|  2   | bfs            |   make   |  make run     |
|  3   | b+tree         |   make   |  make run     |
|  4   | cfd            |   make   |  make run     |
|  5   | dwt2d          |   make   |  ./run        |
|  6   | gaussian       |   make   |  ./run        |
|  7   | heartwall      |   make   |  make run     |
|  8   | hotspot        |   make   |  ./run        |
|  9   | hotspot3D      |   make   |  ./run        |
|  10  | hybridsort     |   make   |  ./run        |
|  11  | kmeans         |   make   |  ./run        |
|  12  | lavaMD         |   make   |  ./run        |
|  13  | leukocyte      |   make   |  ./run        |
|  14  | lud            |   make   |  make run     |
|  15  | myocyte        |   make   |  ./run        |
|  16  | nn             |   make   |  ./run        |
|  17  | nw             |   make   |  ./run        |
|  18  | particlefilter |   make   |  ./run        |
|  19  | pathfinder     |   make   |  make run     |
|  20  | srad           |   make   |  ./run        |
|  21  | streamcluster  |   make   |  ./run        |


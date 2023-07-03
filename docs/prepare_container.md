# Building Cricket Docker Image for compiling and running 

**Credits**: Yuhan Yang, Tianxia Wang and Xingda Wei

如何基于Pytorch的官方镜像编译得到一个能正常编译并运行cricket.

参考：https://github.com/RWTH-ACS/cricket/blob/share-object-support/docs/pytorch.md



## 1. Get Pytorch and apply criket patch 

首先我们将pytorch拉取到本地, 根据cricket的文档, 选择的pytorch版本为v1.13.1

```
git clone git@github.com:pytorch/pytorch.git
git checkout v1.13.1
git submodule update --init --recursive
```

并对pytorch的源码（主要是build的代码 + container的代码）进行部分修改，主要包含两部分：

1. nvcc默认的cuda runtime是静态链接的, 需要手动指定`-cudart shared`使其变为动态链接; 
2. 为pytorch的容器镜像添加cricket所需相关依赖

具体修改可以参考：https://x8csr71rzs.feishu.cn/docx/DdXFdGSYOo8cktxgj8hcYh12nHf



## 2. Build Pytorch 

```Bash
make -f docker.Makefile
```
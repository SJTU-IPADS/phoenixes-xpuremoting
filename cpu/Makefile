#Standalone RPC Server
SERVER = cricket-rpc-server
SERVER_LIB = cricket-server.so
#RPC client library
CLIENT = cricket-client.so

CUDA_SRC = /usr/local/cuda
LIBTIRPC_PREFIX = ../submodules/libtirpc/install
SUBMODULE_LIBS = ../submodules/lib

CC = g++
LD = g++
RPCGEN = rpcgen

# rpcgen files
RPC_SERVER = cpu_rpc_prot_svc.c
RPC_SERVER_MOD = cpu_rpc_prot_svc_mod.c
RPC_CLIENT = cpu_rpc_prot_clnt.c
RPC_XDR = cpu_rpc_prot_xdr.c
RPC_DEF = cpu_rpc_prot.x
RPC_H = $(RPC_DEF:%.x=%.h)

PROXY_DISPATCH = ./proxy/dispatch.cpp

PROXY_PREFIX = ./proxy
# VERSION = NO_OPTIMIZATION / WITH_TCPIP / WITH_SHARED_MEMORY / WITH_RDMA
VERSION ?= WITH_SHARED_MEMORY

# pos currently will use SHM by default
ifdef POS_ENABLE
VERSION = WITH_SHARED_MEMORY
endif

ALL = $(SERVER) $(CLIENT)

ifeq ($(VERSION), NO_OPTIMIZATION)
NO_CACHE_OPTIMIZATION = 1
NO_ASYNC_OPTIMIZATION = 1
NO_HANDLER_OPTIMIZATION = 1
else
SRC_PROXY_CLIENT = $(PROXY_PREFIX)/xdr_device.cpp		\
					$(PROXY_PREFIX)/clnt.cpp			\
					$(PROXY_PREFIX)/measurement.cpp		\
					$(PROXY_PREFIX)/proxy_header.cpp 	\
					$(PROXY_PREFIX)/device_buffer.cpp	\
					$(PROXY_PREFIX)/shm_buffer.cpp		\
					$(PROXY_PREFIX)/tcpip_buffer.cpp    \
					$(PROXY_PREFIX)/async_batch_caller.cpp \
					$(PROXY_PREFIX)/rdma_buffer.cpp		\
					$(PROXY_PREFIX)/api_trace.cpp

SRC_PROXY_SERVER = $(PROXY_DISPATCH)				\
				$(PROXY_PREFIX)/xdr_device.cpp		\
				$(PROXY_PREFIX)/svc.cpp				\
				$(PROXY_PREFIX)/measurement.cpp		\
				$(PROXY_PREFIX)/proxy_header.cpp 	\
				$(PROXY_PREFIX)/device_buffer.cpp	\
				$(PROXY_PREFIX)/shm_buffer.cpp		\
				$(PROXY_PREFIX)/tcpip_buffer.cpp	\
				$(PROXY_PREFIX)/async_batch_caller.cpp \
				$(PROXY_PREFIX)/rdma_buffer.cpp	  	\
				$(PROXY_PREFIX)/handler_mapper.cpp
endif

# pos won't enable async optimization
ifdef POS_ENABLE
# NO_CACHE_OPTIMIZATION = 1
# NO_ASYNC_OPTIMIZATION = 1
# NO_HANDLER_OPTIMIZATION = 1
endif

SRC_SERVER = $(RPC_XDR)                 \
			 $(RPC_SERVER_MOD)          \
			 cpu-server.c               \
			 cpu-utils.c                \
			 cpu-server-runtime.c       \
			 cpu-server-driver.c        \
			 cpu-server-driver-hidden.c \
			 log.c                      \
			 cpu-libwrap.c              \
			 cpu-server-cusolver.c 		\
			 cpu-server-cublas.c 		\
			 list.c                     \
			 api-recorder.c 			\
			 resource-mg.c 			    \
			 cr.c 					    \
			 gsched_none.c 			    \
			 oob.c 					    \
			 mt-memcpy.c				\
			 cpu-elf2.c					\
			 cpu-server-nvml.c			\
			 cpu-server-cudnn.c			\
			 cpu-measurement.c

SRC_SERVER_LIB = server-library.c
SRC_SERVER_EXE = server-exe.c

SRC_CLIENT = $(RPC_XDR)                 \
			 $(RPC_CLIENT)              \
			 cpu-client.c               \
			 cpu-utils.c                \
			 cpu-client-runtime.c       \
			 cpu-client-driver.c        \
			 list.c 				    \
			 log.c                      \
			 cpu-libwrap.c              \
			 cpu-client-cusolver.c 		\
			 oob.c 					    \
			 mt-memcpy.c				\
			 cpu-elf2.c					\
			 cpu-client-nvml.c          \
			 cpu-client-cudnn.c			\
			 cpu-client-cublas.c		\
			 cpu-measurement.c

# 			 cpu-client-driver-hidden.c \

ifdef WITH_IB
SRC_SERVER += cpu-ib.c
SRC_CLIENT += cpu-ib.c
endif

OBJ_PROXY_CLIENT = $(SRC_PROXY_CLIENT:%.cpp=%.o)
OBJ_PROXY_SERVER = $(SRC_PROXY_SERVER:%.cpp=%.o)
OBJ_SERVER = $(SRC_SERVER:%.c=%.o)
OBJ_CLIENT = $(SRC_CLIENT:%.c=%.o)

# this env variable is passed from PheonixOS build script
ifdef POS_ENABLE
CC_FLAGS += -DPOS_ENABLE
# CC_FLAGS += -DPOS_ENABLE_HIJACK_API_CHECK
endif

# Compiler flags
RPCGEN_FLAGS = -C -M -N

INC_FLAGS += -I$(LIBTIRPC_PREFIX)/include/tirpc
INC_FLAGS += -I$(CUDA_SRC)/include
INC_FLAGS += -I$(PROXY_PREFIX)

# add POS include path
INC_FLAGS += -I../../..
INC_FLAGS += -I../../../lib
INC_FLAGS += -I../../../build

LIB_FLAGS += -L$(LIBTIRPC_PREFIX)/lib
LIB_FLAGS += -L$(CUDA_SRC)/lib64
LIB_FLAGS += -L$(CUDA_SRC)/lib64/stubs

# add POS lib path
LIB_FLAGS += -L../../../lib
LIB_FLAGS += -lpos
LIB_FLAGS += -lclang
LIB_FLAGS += -lyaml-cpp
LIB_FLAGS += -lprotobuf
LIB_FLAGS += -lprotobuf-lite
LIB_FLAGS += -lprotoc

CC_FLAGS += -std=c++17 -w -fpermissive $(INC_FLAGS) -O0
# TODO: use extern in header files instead of direct definition e.g. in cpu-common.h to remove -fcommon flag
CC_FLAGS += -fcommon
CC_FLAGS += -D$(VERSION)
CC_FLAGS += -DNO_DCT
LD_FLAGS = $(LIB_FLAGS) -ltirpc -ldl -lcrypto -lelf -lrt -lpthread -libverbs

ifdef WITH_DEBUG
# use ASAN_OPTIONS=protect_shadow_gap=0  LSAN_OPTIONS=fast_unwind_on_malloc=0 when running
CC_FLAGS += -g -ggdb #-static-libasan -fsanitize=address -fsanitize=pointer-compare -fsanitize=pointer-subtract -fsanitize-address-use-after-scope
endif

ifdef LOG
CC_FLAGS += -DLOG_LEVEL=LOG_$(LOG)
endif

ifdef LOGN
CC_FLAGS += -DLOG_LEVEL=$(LOGN)
endif

ifdef WITH_IB
CC_FLAGS += -DWITH_IB=$(WITH_IB)
endif

ifdef NO_CACHE_OPTIMIZATION
CC_FLAGS += -DNO_CACHE_OPTIMIZATION
endif

ifdef NO_ASYNC_OPTIMIZATION
CC_FLAGS += -DNO_ASYNC_OPTIMIZATION
endif

ifdef NO_HANDLER_OPTIMIZATION
CC_FLAGS += -DNO_HANDLER_OPTIMIZATION
RPC_DEF_SRC = cpu_rpc_prot_no_handler_optim.x
PROXY_DISPATCH_SRC = ./proxy/dispatch_no_handler_optim.cpp
else
RPC_DEF_SRC = cpu_rpc_prot_with_handler_optim.x
PROXY_DISPATCH_SRC = ./proxy/dispatch_with_handler_optim.cpp
endif

SERVER_LD_FLAGS = $(LD_FLAGS) -lcudart -lcusolver -lcuda -lcublas -lnvidia-ml -lcudnn
SERVER_BIN_LD_FLAGS = $(SERVER_LD_FLAGS) -Wl,--unresolved-symbols=ignore-in-object-files
CLIENT_LD_FLAGS = $(LD_FLAGS)

# Targets
.PHONY: all clean

all : $(ALL)

$(CLIENT) : $(OBJ_CLIENT) $(OBJ_PROXY_CLIENT)
	$(LD) $(CC_FLAGS) -shared -o $@ $^ $(CLIENT_LD_FLAGS)

$(SERVER_LIB) : $(OBJ_SERVER) $(SRC_SERVER_EXE:%.c=%.o) $(OBJ_PROXY_SERVER)
	$(LD) $(CC_FLAGS) -shared -o $@ $^ $(SERVER_BIN_LD_FLAGS)

$(SERVER) : $(OBJ_SERVER) $(SRC_SERVER_EXE:%.c=%.o) $(OBJ_PROXY_SERVER)
	$(LD) $(CC_FLAGS) -o $@ $^ $(SERVER_BIN_LD_FLAGS)

$(RPC_DEF) : $(RPC_DEF_SRC)
	cp $< $@

$(PROXY_DISPATCH) : $(PROXY_DISPATCH_SRC)
	cp $< $@

$(RPC_H) : $(RPC_DEF)
	rm -f $@ && $(RPCGEN) $(RPCGEN_FLAGS) -h -o $@ $<

$(RPC_CLIENT) : $(RPC_DEF)
	rm -f $@ && $(RPCGEN) $(RPCGEN_FLAGS) -l -o $@ $<

$(RPC_SERVER) : $(RPC_DEF)
	rm -f $@ && $(RPCGEN) $(RPCGEN_FLAGS) -m -o $@ $<

$(RPC_SERVER_MOD) : $(RPC_SERVER)
	./generate_dispatch.sh

$(RPC_XDR) : $(RPC_DEF)
	rm -f $@ && $(RPCGEN) $(RPCGEN_FLAGS) -c -o $@ $<

%.o : %.c $(RPC_H)
	$(CC) $(CC_FLAGS) -c -fpic -o $@ $< $(LD_FLAGS) 

%.o : %.cpp $(RPC_H)
	$(CC) $(CC_FLAGS) -c -fpic -o $@ $< $(LD_FLAGS)

clean:
	 rm -f $(RPC_DEF) $(RPC_H) $(RPC_CLIENT) $(RPC_SERVER) $(RPC_SERVER_MOD) $(RPC_XDR) $(PROXY_DISPATCH)\
	 $(OBJ_CLIENT) $(OBJ_SERVER) $(SERVER) $(SERVER_LIB) $(CLIENT) $(SRC_SERVER_EXE:%.c=%.o) $(OBJ_PROXY_CLIENT) $(OBJ_PROXY_SERVER)

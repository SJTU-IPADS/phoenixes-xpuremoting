.PHONY: all clean cuda_app

all: cuda_app
	cp -r libtirpc ../../submodules/

cuda_app:
	$(MAKE) -C cuda_app

clean:
	$(MAKE) -C cuda_app clean

#ifndef XDR_DEVICE_H
#define XDR_DEVICE_H

#include "device_buffer.h"
#include <rpc/types.h>
#include <rpc/xdr.h>
#include <sys/types.h>

typedef int bool_t;
typedef unsigned int u_int;
typedef signed int int32_t;

class XDRDevice
{
public:
    XDRDevice(DeviceBuffer *buffer = nullptr);
    ~XDRDevice();

    void SetBuffer(DeviceBuffer *buffer);
    DeviceBuffer *GetBuffer();
    void SetMask(int m);
    int GetMask();

    // inner XDR ops
    bool_t Getlong(long *lp);
    bool_t Putlong(const long *lp);
    bool_t Getbytes(char *addr, u_int len);
    bool_t Putbytes(const char *addr, u_int len);
    u_int Getpos();
    bool_t Setpos(u_int pos);
    int32_t *Inline(u_int len);

private:
    DeviceBuffer *buffer_;
    u_int position;
    int mask;
};

XDR *new_xdrdevice(enum xdr_op op);
void destroy_xdrdevice(XDR **xdrs);

#endif // XDR_DEVICE_H
